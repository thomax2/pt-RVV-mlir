#include "eot/EotPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <limits>

namespace eot {
#define GEN_PASS_DEF_PLANSTATICWORKSPACE
#include "eot/EotPasses.h.inc"
} // namespace eot

using namespace mlir;

namespace {
// This pass operates only on Arith, Func, and MemRef IR. It belongs to the EOT
// compilation pipeline but has no dependency on the EOT or legacy NPU dialect.
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
      for (memref::AllocOp alloc : allocations) {
        MemRefType type = alloc.getType();
        if (!type.hasStaticShape()) {
          alloc.emitOpError("static workspace requires a static memref");
          signalPassFailure();
          return;
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
          signalPassFailure();
          return;
        }
        if (bitWidth == 0) {
          alloc.emitOpError(
              "static workspace requires an element type with a known width");
          signalPassFailure();
          return;
        }
        if (!type.getLayout().isIdentity()) {
          alloc.emitOpError(
              "static workspace requires an identity-layout memref");
          signalPassFailure();
          return;
        }
        if (type.getMemorySpace()) {
          alloc.emitOpError(
              "static workspace requires the default memory space");
          signalPassFailure();
          return;
        }
        if (auto alignment =
                alloc->getAttrOfType<IntegerAttr>("alignment")) {
          int64_t value = alignment.getInt();
          if (value <= 0 || !llvm::isPowerOf2_64(value)) {
            alloc.emitOpError(
                "alignment must be a positive power of two in bytes");
            signalPassFailure();
            return;
          }
          requiredWorkspaceAlignment =
              std::max(requiredWorkspaceAlignment, value);
        }
      }

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

      int64_t currentOffsetBytes = 0;
      for (memref::AllocOp alloc : allocations) {
        MemRefType type = alloc.getType();
        Type elementType = type.getElementType();
        int64_t bitWidth = isa<IntegerType>(elementType)
                               ? cast<IntegerType>(elementType).getWidth()
                               : cast<FloatType>(elementType).getWidth();
        // LLVM allocates sub-byte integer elements in addressable byte units.
        int64_t elementBytes = (bitWidth + 7) / 8;
        int64_t allocationAlignment = 16;
        if (auto alignment =
                alloc->getAttrOfType<IntegerAttr>("alignment"))
          allocationAlignment = std::max(allocationAlignment,
                                         alignment.getInt());
        if (currentOffsetBytes >
            std::numeric_limits<int64_t>::max() -
                (allocationAlignment - 1)) {
          alloc.emitOpError("static workspace offset overflow");
          signalPassFailure();
          return;
        }
        currentOffsetBytes =
            llvm::alignTo(currentOffsetBytes, allocationAlignment);

        int64_t elementCount = 1;
        for (int64_t dim : type.getShape()) {
          if (dim != 0 &&
              elementCount > std::numeric_limits<int64_t>::max() / dim) {
            alloc.emitOpError("static workspace element-count overflow");
            signalPassFailure();
            return;
          }
          elementCount *= dim;
        }
        if (elementCount != 0 &&
            elementBytes >
                std::numeric_limits<int64_t>::max() / elementCount) {
          alloc.emitOpError("static workspace byte-size overflow");
          signalPassFailure();
          return;
        }
        int64_t allocationBytes = elementCount * elementBytes;
        if (allocationBytes >
            std::numeric_limits<int64_t>::max() - currentOffsetBytes) {
          alloc.emitOpError("static workspace size overflow");
          signalPassFailure();
          return;
        }

        builder.setInsertionPoint(alloc);
        Value byteShift = builder.create<arith::ConstantIndexOp>(
            alloc.getLoc(), currentOffsetBytes);
        Value view = builder.create<memref::ViewOp>(
            alloc.getLoc(), type, workspace, byteShift, ValueRange{});
        alloc.replaceAllUsesWith(view);
        alloc.erase();
        currentOffsetBytes += allocationBytes;
      }
      function->setAttr("eot.workspace_bytes",
                        builder.getI64IntegerAttr(currentOffsetBytes));
      function->setAttr(
          "eot.workspace_alignment",
          builder.getI64IntegerAttr(requiredWorkspaceAlignment));
      llvm::outs() << "[eot-plan-static-workspace] " << function.getName()
                   << ": " << currentOffsetBytes << " bytes, alignment "
                   << requiredWorkspaceAlignment << " bytes\n";
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass> eot::createPlanStaticWorkspacePass() {
  return std::make_unique<PlanStaticWorkspacePass>();
}
