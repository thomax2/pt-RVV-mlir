#include "eot/EotPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <string>
#include <utility>

namespace eot {
#define GEN_PASS_DEF_PLANSTATICWORKSPACE
#include "eot/EotPasses.h.inc"
} // namespace eot

using namespace mlir;

namespace {

/// A single independently allocated workspace object.
///
/// Lifetime is a half-open interval [start, end). Objects whose lifetimes do
/// not overlap may share byte ranges. If analysis cannot prove that reuse is
/// safe, reusable is cleared and the object is conservatively made live for
/// the whole function.
struct MemoryObject {
  unsigned id = 0;
  memref::AllocOp alloc;
  int64_t sizeBytes = 0;
  int64_t alignment = 1;
  int64_t start = 0;
  int64_t end = 0;
  Value aliasRoot;
  Attribute memorySpace;
  bool reusable = true;
  std::string nonReusableReason;

  int64_t offset = 0;
  bool reused = false;
};

struct FreeBlock {
  int64_t offset = 0;
  int64_t size = 0;
};

static bool checkedAdd(int64_t lhs, int64_t rhs, int64_t &result) {
  if (lhs < 0 || rhs < 0 ||
      lhs > std::numeric_limits<int64_t>::max() - rhs)
    return false;
  result = lhs + rhs;
  return true;
}

static bool checkedAlignTo(int64_t value, int64_t alignment,
                           int64_t &result) {
  if (value < 0 || alignment <= 0 ||
      value > std::numeric_limits<int64_t>::max() - (alignment - 1))
    return false;
  result = llvm::alignTo(value, alignment);
  return true;
}

static bool byteRangesOverlap(const MemoryObject &lhs,
                              const MemoryObject &rhs) {
  if (lhs.sizeBytes == 0 || rhs.sizeBytes == 0)
    return false;
  int64_t lhsEnd = lhs.offset + lhs.sizeBytes;
  int64_t rhsEnd = rhs.offset + rhs.sizeBytes;
  return lhs.offset < rhsEnd && rhs.offset < lhsEnd;
}

static bool lifetimesOverlap(const MemoryObject &lhs,
                             const MemoryObject &rhs) {
  return lhs.start < rhs.end && rhs.start < lhs.end;
}

static bool isKnownAliasProducer(Operation *operation, Value source) {
  if (operation->getNumOperands() == 0 || operation->getOperand(0) != source)
    return false;

  // These operations create metadata-only views of their first operand.
  // Keeping the list explicit makes the analysis conservative: a new or
  // unfamiliar memref-producing operation is pinned instead of guessed.
  StringRef name = operation->getName().getStringRef();
  return name == "memref.assume_alignment" || name == "memref.cast" ||
         name == "memref.collapse_shape" ||
         name == "memref.expand_shape" ||
         name == "memref.extract_strided_metadata" ||
         name == "memref.memory_space_cast" ||
         name == "memref.reinterpret_cast" || name == "memref.reshape" ||
         name == "memref.subview" || name == "memref.transpose" ||
         name == "memref.view";
}

static bool isLoopLike(Operation *operation) {
  StringRef name = operation->getName().getStringRef();
  return name == "scf.for" || name == "scf.while" ||
         name == "scf.parallel" || name == "scf.forall" ||
         name == "affine.for" || name == "affine.parallel";
}

static bool isAsynchronous(Operation *operation) {
  StringRef name = operation->getName().getStringRef();
  return name.starts_with("async.") || name == "gpu.launch" ||
         name == "gpu.launch_func";
}

static bool isKnownSynchronousDialect(Operation *operation) {
  StringRef dialect =
      operation->getName().getStringRef().split('.').first;
  return dialect == "affine" || dialect == "arith" || dialect == "cf" ||
         dialect == "func" || dialect == "index" || dialect == "linalg" ||
         dialect == "math" || dialect == "memref" || dialect == "scf" ||
         dialect == "ub" || dialect == "vector";
}

static void markNonReusable(MemoryObject &object, StringRef reason) {
  if (!object.reusable)
    return;
  object.reusable = false;
  object.nonReusableReason = reason.str();
}

static bool computeObjectSizeAndAlignment(memref::AllocOp alloc,
                                          MemoryObject &object,
                                          int64_t &workspaceAlignment) {
  MemRefType type = alloc.getType();
  if (!type.hasStaticShape()) {
    alloc.emitOpError("static workspace requires a static memref");
    return false;
  }
  if (!type.getLayout().isIdentity()) {
    alloc.emitOpError("static workspace requires an identity-layout memref");
    return false;
  }
  if (type.getMemorySpace()) {
    alloc.emitOpError("static workspace requires the default memory space");
    return false;
  }

  Type elementType = type.getElementType();
  int64_t bitWidth = 0;
  if (auto integerType = dyn_cast<IntegerType>(elementType))
    bitWidth = integerType.getWidth();
  else if (auto floatType = dyn_cast<FloatType>(elementType))
    bitWidth = floatType.getWidth();
  else {
    alloc.emitOpError(
        "static workspace supports only integer and floating-point "
        "element types");
    return false;
  }
  if (bitWidth == 0) {
    alloc.emitOpError(
        "static workspace requires an element type with a known width");
    return false;
  }

  int64_t alignment = 16;
  if (auto alignmentAttr = alloc->getAttrOfType<IntegerAttr>("alignment")) {
    int64_t value = alignmentAttr.getInt();
    if (value <= 0 || !llvm::isPowerOf2_64(value)) {
      alloc.emitOpError("alignment must be a positive power of two in bytes");
      return false;
    }
    alignment = std::max(alignment, value);
  }

  int64_t elementCount = 1;
  for (int64_t dim : type.getShape()) {
    if (dim != 0 &&
        elementCount > std::numeric_limits<int64_t>::max() / dim) {
      alloc.emitOpError("static workspace element-count overflow");
      return false;
    }
    elementCount *= dim;
  }

  // Sub-byte integer elements still occupy addressable byte units here.
  int64_t elementBytes = (bitWidth + 7) / 8;
  if (elementCount != 0 &&
      elementBytes > std::numeric_limits<int64_t>::max() / elementCount) {
    alloc.emitOpError("static workspace byte-size overflow");
    return false;
  }

  object.alloc = alloc;
  object.sizeBytes = elementCount * elementBytes;
  object.alignment = alignment;
  object.aliasRoot = alloc.getResult();
  object.memorySpace = type.getMemorySpace();
  workspaceAlignment = std::max(workspaceAlignment, alignment);
  return true;
}

/// Assigns a stable integer position to every operation and records the last
/// position in each operation subtree. The latter lets loop uses conservatively
/// cover the complete loop body.
static int64_t
numberOperations(func::FuncOp function,
                 DenseMap<Operation *, int64_t> &operationPosition,
                 DenseMap<Operation *, int64_t> &subtreeEnd) {
  int64_t nextPosition = 0;
  function.walk<WalkOrder::PreOrder>([&](Operation *operation) {
    int64_t position = nextPosition++;
    operationPosition[operation] = position;
    for (Operation *ancestor = operation; ancestor != nullptr;
         ancestor = ancestor->getParentOp()) {
      subtreeEnd[ancestor] = position;
      if (ancestor == function.getOperation())
        break;
    }
  });
  return nextPosition;
}

static void includeUsePosition(
    func::FuncOp function, Operation *user, int64_t &firstUse,
    int64_t &lastUse, const DenseMap<Operation *, int64_t> &operationPosition,
    const DenseMap<Operation *, int64_t> &subtreeEnd) {
  auto positionIt = operationPosition.find(user);
  if (positionIt == operationPosition.end())
    return;
  firstUse = std::min(firstUse, positionIt->second);
  lastUse = std::max(lastUse, positionIt->second + 1);

  // A use in a loop may execute again after every syntactically later use in
  // that loop. Expanding to the loop subtree avoids an unsafe early release.
  for (Operation *ancestor = user; ancestor != nullptr;
       ancestor = ancestor->getParentOp()) {
    if (ancestor == function.getOperation())
      break;
    if (!isLoopLike(ancestor))
      continue;
    auto startIt = operationPosition.find(ancestor);
    auto endIt = subtreeEnd.find(ancestor);
    if (startIt != operationPosition.end() && endIt != subtreeEnd.end()) {
      firstUse = std::min(firstUse, startIt->second);
      lastUse = std::max(lastUse, endIt->second + 1);
    }
  }
}

/// Computes an alias closure and a conservative lifetime for one allocation.
/// Unknown calls, region forwarding, CFG forwarding, or memref-producing uses
/// pin the object for the full function instead of risking an invalid reuse.
static void analyzeObjectLifetime(
    func::FuncOp function, MemoryObject &object, int64_t operationCount,
    const DenseMap<Operation *, int64_t> &operationPosition,
    const DenseMap<Operation *, int64_t> &subtreeEnd) {
  int64_t firstUse = std::numeric_limits<int64_t>::max();
  int64_t lastUse = 0;
  DenseSet<Value> aliases;
  SmallVector<Value> worklist;
  aliases.insert(object.aliasRoot);
  worklist.push_back(object.aliasRoot);

  if (object.alloc->getBlock() != &function.front())
    markNonReusable(object, "allocation is not in the entry block");

  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    for (OpOperand &use : value.getUses()) {
      Operation *user = use.getOwner();

      if (isKnownAliasProducer(user, value)) {
        bool foundAliasResult = false;
        for (Value result : user->getResults()) {
          if (!isa<BaseMemRefType>(result.getType()))
            continue;
          foundAliasResult = true;
          if (aliases.insert(result).second)
            worklist.push_back(result);
        }
        if (foundAliasResult)
          continue;
      }

      includeUsePosition(function, user, firstUse, lastUse,
                         operationPosition, subtreeEnd);

      if (isa<func::CallOp>(user)) {
        markNonReusable(object, "passed to func.call");
        continue;
      }

      StringRef userName = user->getName().getStringRef();
      if (userName == "func.call_indirect") {
        markNonReusable(object, "passed to func.call_indirect");
        continue;
      }
      if (userName == "scf.yield" || userName == "scf.condition" ||
          user->getNumSuccessors() != 0) {
        markNonReusable(object, "forwarded through region or CFG control flow");
        continue;
      }
      for (Operation *ancestor = user; ancestor != nullptr;
           ancestor = ancestor->getParentOp()) {
        if (isAsynchronous(ancestor)) {
          markNonReusable(object, "used by asynchronous execution");
          break;
        }
        if (!isKnownSynchronousDialect(ancestor)) {
          markNonReusable(object, "used by an unknown dialect");
          break;
        }
        if (ancestor == function.getOperation())
          break;
      }

      if (llvm::any_of(user->getResultTypes(), [](Type type) {
            return isa<BaseMemRefType>(type);
          })) {
        markNonReusable(object,
                        "unknown memref-producing use: " + userName.str());
      }
    }
  }

  if (!object.reusable) {
    object.start = 0;
    object.end = operationCount + 1;
    return;
  }

  if (firstUse == std::numeric_limits<int64_t>::max()) {
    int64_t allocationPosition =
        operationPosition.lookup(object.alloc.getOperation());
    object.start = allocationPosition;
    object.end = allocationPosition + 1;
    return;
  }

  // Bufferization frequently hoists memref.alloc operations. Starting at the
  // first real consumer, rather than the alloc itself, exposes safe reuse.
  object.start = firstUse;
  object.end = std::max(firstUse + 1, lastUse);
}

static void addFreeBlock(SmallVectorImpl<FreeBlock> &freeBlocks,
                         int64_t offset, int64_t size) {
  if (size <= 0)
    return;
  freeBlocks.push_back({offset, size});
  llvm::sort(freeBlocks, [](const FreeBlock &lhs, const FreeBlock &rhs) {
    return lhs.offset < rhs.offset;
  });

  SmallVector<FreeBlock> merged;
  for (const FreeBlock &block : freeBlocks) {
    if (merged.empty()) {
      merged.push_back(block);
      continue;
    }
    FreeBlock &previous = merged.back();
    int64_t previousEnd = previous.offset + previous.size;
    int64_t blockEnd = block.offset + block.size;
    if (block.offset <= previousEnd) {
      previous.size = std::max(previousEnd, blockEnd) - previous.offset;
      continue;
    }
    merged.push_back(block);
  }
  freeBlocks.assign(merged.begin(), merged.end());
}

static bool buildSequentialPlan(MutableArrayRef<MemoryObject> objects,
                                int64_t &workspaceBytes) {
  workspaceBytes = 0;
  for (MemoryObject &object : objects) {
    int64_t alignedOffset = 0;
    if (!checkedAlignTo(workspaceBytes, object.alignment, alignedOffset))
      return false;
    int64_t nextOffset = 0;
    if (!checkedAdd(alignedOffset, object.sizeBytes, nextOffset))
      return false;
    object.offset = alignedOffset;
    object.reused = false;
    workspaceBytes = nextOffset;
  }
  return true;
}

/// Deterministic scan-line Best-Fit. Objects are introduced by lifetime start;
/// expired ranges enter a coalesced free list. Ties are resolved by lower byte
/// offset and then object id.
static bool buildBestFitPlan(MutableArrayRef<MemoryObject> objects,
                             int64_t &workspaceBytes) {
  SmallVector<unsigned> order;
  order.reserve(objects.size());
  for (unsigned index = 0; index < objects.size(); ++index) {
    order.push_back(index);
    objects[index].offset = 0;
    objects[index].reused = false;
  }
  llvm::sort(order, [&](unsigned lhsIndex, unsigned rhsIndex) {
    const MemoryObject &lhs = objects[lhsIndex];
    const MemoryObject &rhs = objects[rhsIndex];
    if (lhs.start != rhs.start)
      return lhs.start < rhs.start;
    if (lhs.sizeBytes != rhs.sizeBytes)
      return lhs.sizeBytes > rhs.sizeBytes;
    return lhs.id < rhs.id;
  });

  SmallVector<unsigned> active;
  SmallVector<FreeBlock> freeBlocks;
  workspaceBytes = 0;

  for (unsigned objectIndex : order) {
    MemoryObject &object = objects[objectIndex];

    SmallVector<unsigned> stillActive;
    for (unsigned activeIndex : active) {
      MemoryObject &activeObject = objects[activeIndex];
      if (activeObject.end <= object.start) {
        addFreeBlock(freeBlocks, activeObject.offset,
                     activeObject.sizeBytes);
      } else {
        stillActive.push_back(activeIndex);
      }
    }
    active.swap(stillActive);

    std::optional<unsigned> bestBlockIndex;
    int64_t bestAlignedOffset = 0;
    int64_t bestWaste = std::numeric_limits<int64_t>::max();
    for (unsigned blockIndex = 0; blockIndex < freeBlocks.size();
         ++blockIndex) {
      const FreeBlock &block = freeBlocks[blockIndex];
      int64_t alignedOffset = 0;
      if (!checkedAlignTo(block.offset, object.alignment, alignedOffset))
        continue;
      int64_t blockEnd = block.offset + block.size;
      if (alignedOffset > blockEnd ||
          object.sizeBytes > blockEnd - alignedOffset)
        continue;
      int64_t waste = block.size - object.sizeBytes;
      if (!bestBlockIndex || waste < bestWaste ||
          (waste == bestWaste && alignedOffset < bestAlignedOffset)) {
        bestBlockIndex = blockIndex;
        bestAlignedOffset = alignedOffset;
        bestWaste = waste;
      }
    }

    if (bestBlockIndex) {
      FreeBlock selected = freeBlocks[*bestBlockIndex];
      freeBlocks.erase(freeBlocks.begin() + *bestBlockIndex);
      object.offset = bestAlignedOffset;
      addFreeBlock(freeBlocks, selected.offset,
                   bestAlignedOffset - selected.offset);
      int64_t objectEnd = bestAlignedOffset + object.sizeBytes;
      int64_t selectedEnd = selected.offset + selected.size;
      addFreeBlock(freeBlocks, objectEnd, selectedEnd - objectEnd);
    } else {
      int64_t alignedOffset = 0;
      if (!checkedAlignTo(workspaceBytes, object.alignment, alignedOffset))
        return false;
      addFreeBlock(freeBlocks, workspaceBytes,
                   alignedOffset - workspaceBytes);
      int64_t nextOffset = 0;
      if (!checkedAdd(alignedOffset, object.sizeBytes, nextOffset))
        return false;
      object.offset = alignedOffset;
      workspaceBytes = nextOffset;
    }
    active.push_back(objectIndex);
  }
  return true;
}

static bool verifyPlan(ArrayRef<MemoryObject> objects, int64_t workspaceBytes,
                       std::string &reason) {
  for (const MemoryObject &object : objects) {
    if (object.offset < 0 || object.alignment <= 0 ||
        object.offset % object.alignment != 0) {
      reason = "object #" + std::to_string(object.id) +
               " has an invalid or misaligned offset";
      return false;
    }
    if (object.offset > workspaceBytes ||
        object.sizeBytes > workspaceBytes - object.offset) {
      reason = "object #" + std::to_string(object.id) +
               " exceeds the workspace boundary";
      return false;
    }
  }

  for (unsigned lhsIndex = 0; lhsIndex < objects.size(); ++lhsIndex) {
    for (unsigned rhsIndex = lhsIndex + 1; rhsIndex < objects.size();
         ++rhsIndex) {
      const MemoryObject &lhs = objects[lhsIndex];
      const MemoryObject &rhs = objects[rhsIndex];
      if (lifetimesOverlap(lhs, rhs) && byteRangesOverlap(lhs, rhs)) {
        reason = "live objects #" + std::to_string(lhs.id) + " and #" +
                 std::to_string(rhs.id) + " overlap in the workspace";
        return false;
      }
    }
  }
  return true;
}

static int64_t computePeakLiveBytes(ArrayRef<MemoryObject> objects) {
  SmallVector<std::pair<int64_t, int64_t>> events;
  events.reserve(objects.size() * 2);
  for (const MemoryObject &object : objects) {
    if (object.sizeBytes == 0)
      continue;
    events.push_back({object.start, object.sizeBytes});
    events.push_back({object.end, -object.sizeBytes});
  }
  llvm::sort(events, [](const auto &lhs, const auto &rhs) {
    if (lhs.first != rhs.first)
      return lhs.first < rhs.first;
    // End events precede start events for half-open lifetimes.
    return lhs.second < rhs.second;
  });

  int64_t liveBytes = 0;
  int64_t peakBytes = 0;
  for (const auto &[time, delta] : events) {
    (void)time;
    liveBytes += delta;
    peakBytes = std::max(peakBytes, liveBytes);
  }
  return peakBytes;
}

static unsigned markAndCountReusedObjects(MutableArrayRef<MemoryObject> objects) {
  unsigned reusedCount = 0;
  for (unsigned index = 0; index < objects.size(); ++index) {
    MemoryObject &object = objects[index];
    object.reused = false;
    for (unsigned previous = 0; previous < index; ++previous) {
      if (!lifetimesOverlap(object, objects[previous]) &&
          byteRangesOverlap(object, objects[previous])) {
        object.reused = true;
        break;
      }
    }
    reusedCount += object.reused ? 1 : 0;
  }
  return reusedCount;
}

// This pass operates only on Arith, Func, and MemRef IR. It belongs to the EOT
// compilation pipeline but has no dependency on the EOT dialect itself.
struct PlanStaticWorkspacePass
    : ::eot::impl::PlanStaticWorkspaceBase<PlanStaticWorkspacePass> {
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() final {
    OpBuilder builder(&getContext());
    for (func::FuncOp function : getOperation().getOps<func::FuncOp>()) {
      if (function.isExternal())
        continue;

      SmallVector<memref::AllocOp> allocations;
      SmallVector<memref::AllocaOp> stackAllocations;
      SmallVector<memref::DeallocOp> deallocations;
      function.walk([&](memref::AllocOp alloc) { allocations.push_back(alloc); });
      function.walk(
          [&](memref::AllocaOp alloc) { stackAllocations.push_back(alloc); });
      function.walk(
          [&](memref::DeallocOp dealloc) { deallocations.push_back(dealloc); });

      if (!stackAllocations.empty()) {
        stackAllocations.front().emitOpError(
            "static workspace does not support memref.alloca");
        signalPassFailure();
        return;
      }
      if (!deallocations.empty()) {
        deallocations.front().emitOpError(
            "static workspace owns planned storage; do not run a deallocation "
            "pipeline before eot-plan-static-workspace");
        signalPassFailure();
        return;
      }
      if (allocations.empty())
        continue;

      bool hasCallers = false;
      getOperation().walk([&](func::CallOp call) {
        if (call.getCallee() == function.getName()) {
          call.emitOpError(
              "static workspace ABI conversion currently supports only "
              "entry/leaf functions without func.call users");
          hasCallers = true;
        }
      });
      if (hasCallers) {
        signalPassFailure();
        return;
      }

      FunctionType oldType = function.getFunctionType();
      SmallVector<Type> oldInputs(oldType.getInputs());
      SmallVector<Type> oldResults(oldType.getResults());
      if (!llvm::all_of(oldResults,
                        [](Type type) { return isa<MemRefType>(type); })) {
        function.emitOpError(
            "static workspace ABI conversion requires every function result "
            "to be a ranked memref");
        signalPassFailure();
        return;
      }

      int64_t requiredWorkspaceAlignment = 16;
      SmallVector<MemoryObject> objects;
      objects.reserve(allocations.size());
      for (auto [id, alloc] : llvm::enumerate(allocations)) {
        MemoryObject object;
        object.id = static_cast<unsigned>(id);
        if (!computeObjectSizeAndAlignment(alloc, object,
                                           requiredWorkspaceAlignment)) {
          signalPassFailure();
          return;
        }
        objects.push_back(std::move(object));
      }

      DenseMap<Operation *, int64_t> operationPosition;
      DenseMap<Operation *, int64_t> subtreeEnd;
      int64_t operationCount =
          numberOperations(function, operationPosition, subtreeEnd);
      for (MemoryObject &object : objects)
        analyzeObjectLifetime(function, object, operationCount,
                              operationPosition, subtreeEnd);

      int64_t sequentialBytes = 0;
      if (!buildSequentialPlan(objects, sequentialBytes)) {
        function.emitOpError("static workspace sequential layout overflow");
        signalPassFailure();
        return;
      }

      int64_t workspaceBytes = 0;
      bool usedFallback = false;
      std::string verificationFailure;
      bool bestFitAccepted = buildBestFitPlan(objects, workspaceBytes);
      if (bestFitAccepted)
        bestFitAccepted =
            verifyPlan(objects, workspaceBytes, verificationFailure);
      if (bestFitAccepted && workspaceBytes > sequentialBytes) {
        verificationFailure =
            "planned size exceeds the sequential baseline";
        bestFitAccepted = false;
      }
      if (!bestFitAccepted) {
        usedFallback = true;
        function.emitWarning()
            << "lifetime-aware Best-Fit plan was rejected"
            << (verificationFailure.empty()
                    ? "; falling back to sequential layout"
                    : ": " + verificationFailure +
                          "; falling back to sequential layout");
        if (!buildSequentialPlan(objects, workspaceBytes) ||
            !verifyPlan(objects, workspaceBytes, verificationFailure)) {
          function.emitOpError()
              << "internal error: sequential workspace verification failed: "
              << verificationFailure;
          signalPassFailure();
          return;
        }
      }

      int64_t peakLiveBytes = computePeakLiveBytes(objects);
      unsigned reusedCount = markAndCountReusedObjects(objects);
      unsigned nonReusableCount =
          llvm::count_if(objects, [](const MemoryObject &object) {
            return !object.reusable;
          });

      auto workspaceType =
          MemRefType::get({ShapedType::kDynamic}, builder.getI8Type());
      SmallVector<Type> newInputs(oldInputs);
      newInputs.append(oldResults);
      newInputs.push_back(workspaceType);
      function.setType(FunctionType::get(&getContext(), newInputs, {}));
      Block &entry = function.front();
      for (Type resultType : oldResults)
        entry.addArgument(resultType, function.getLoc());
      entry.addArgument(workspaceType, function.getLoc());
      Value workspace = entry.getArguments().back();

      SmallVector<func::ReturnOp> returns;
      function.walk([&](func::ReturnOp ret) { returns.push_back(ret); });
      for (func::ReturnOp ret : returns) {
        builder.setInsertionPoint(ret);
        for (auto [index, operand] : llvm::enumerate(ret.getOperands()))
          builder.create<memref::CopyOp>(
              ret.getLoc(), operand,
              entry.getArgument(oldInputs.size() + index));
        builder.create<func::ReturnOp>(ret.getLoc());
        ret.erase();
      }

      for (MemoryObject &object : objects) {
        builder.setInsertionPoint(object.alloc);
        Value byteShift = builder.create<arith::ConstantIndexOp>(
            object.alloc.getLoc(), object.offset);
        Value view = builder.create<memref::ViewOp>(
            object.alloc.getLoc(), object.alloc.getType(), workspace, byteShift,
            ValueRange{});
        object.alloc.replaceAllUsesWith(view);
        object.alloc.erase();
      }

      function->setAttr("eot.workspace_bytes",
                        builder.getI64IntegerAttr(workspaceBytes));
      function->setAttr(
          "eot.workspace_alignment",
          builder.getI64IntegerAttr(requiredWorkspaceAlignment));
      function->setAttr("eot.workspace_sequential_bytes",
                        builder.getI64IntegerAttr(sequentialBytes));
      function->setAttr("eot.workspace_peak_live_bytes",
                        builder.getI64IntegerAttr(peakLiveBytes));
      function->setAttr("eot.workspace_reused_allocations",
                        builder.getI64IntegerAttr(reusedCount));
      function->setAttr("eot.workspace_non_reusable_allocations",
                        builder.getI64IntegerAttr(nonReusableCount));
      function->setAttr(
          "eot.workspace_strategy",
          builder.getStringAttr(usedFallback ? "sequential-fallback"
                                             : "best-fit"));

      llvm::outs() << "[eot-plan-static-workspace] " << function.getName()
                   << ": objects " << objects.size() << ", reused "
                   << reusedCount << ", pinned " << nonReusableCount
                   << ", sequential " << sequentialBytes << " bytes"
                   << ", peak-live lower bound " << peakLiveBytes << " bytes"
                   << ", planned " << workspaceBytes << " bytes"
                   << ", saved " << (sequentialBytes - workspaceBytes)
                   << " bytes, alignment " << requiredWorkspaceAlignment
                   << " bytes, strategy "
                   << (usedFallback ? "sequential-fallback" : "best-fit")
                   << "\n";
      for (const MemoryObject &object : objects) {
        if (!object.reusable)
          llvm::outs() << "  object #" << object.id << " pinned: "
                       << object.nonReusableReason << "\n";
      }
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> eot::createPlanStaticWorkspacePass() {
  return std::make_unique<PlanStaticWorkspacePass>();
}
