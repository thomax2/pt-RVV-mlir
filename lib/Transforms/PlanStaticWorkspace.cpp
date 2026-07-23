#include "npu/NpuPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <limits>

namespace npu {
#define GEN_PASS_DEF_PLANSTATICWORKSPACE
#include "npu/NpuPasses.h.inc"
} // namespace npu

using namespace mlir;

namespace {
// Despite the compatibility command name, this pass operates only on Func and
// MemRef IR and has no dependency on the legacy NPU dialect.
struct PlanStaticWorkspacePass
    : ::npu::impl::PlanStaticWorkspaceBase<PlanStaticWorkspacePass> {
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<func::FuncDialect, memref::MemRefDialect>();
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
            "pipeline before npu-plan-static-workspace");
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
        if (!type.hasStaticShape() || !type.getElementType().isF32()) {
          alloc.emitOpError("static workspace requires a static f32 memref");
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
          if (value <= 0 || !llvm::isPowerOf2_64(value) ||
              value % static_cast<int64_t>(sizeof(float)) != 0) {
            alloc.emitOpError(
                "alignment must be a positive power-of-two multiple of four");
            signalPassFailure();
            return;
          }
          requiredWorkspaceAlignment =
              std::max(requiredWorkspaceAlignment, value);
        }
      }

      auto workspaceType =
          MemRefType::get({ShapedType::kDynamic}, builder.getF32Type());
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

      int64_t currentOffset = 0;
      for (memref::AllocOp alloc : allocations) {
        MemRefType type = alloc.getType();
        int64_t allocationAlignment = 16;
        if (auto alignment =
                alloc->getAttrOfType<IntegerAttr>("alignment"))
          allocationAlignment = std::max(allocationAlignment,
                                         alignment.getInt());
        int64_t alignmentElements =
            allocationAlignment / static_cast<int64_t>(sizeof(float));
        if (currentOffset >
            std::numeric_limits<int64_t>::max() - (alignmentElements - 1)) {
          alloc.emitOpError("static workspace offset overflow");
          signalPassFailure();
          return;
        }
        currentOffset = llvm::alignTo(currentOffset, alignmentElements);

        SmallVector<int64_t> staticStrides(type.getRank());
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
        if (elementCount >
            std::numeric_limits<int64_t>::max() - currentOffset) {
          alloc.emitOpError("static workspace size overflow");
          signalPassFailure();
          return;
        }
        int64_t stride = 1;
        for (int64_t i = type.getRank() - 1; i >= 0; --i) {
          staticStrides[i] = stride;
          stride *= type.getDimSize(i);
        }
        SmallVector<OpFoldResult> sizes, strides;
        for (int64_t dim : type.getShape())
          sizes.push_back(builder.getIndexAttr(dim));
        for (int64_t value : staticStrides)
          strides.push_back(builder.getIndexAttr(value));
        builder.setInsertionPoint(alloc);
        auto layout = StridedLayoutAttr::get(&getContext(), currentOffset,
                                             staticStrides);
        auto viewType = MemRefType::get(type.getShape(), type.getElementType(),
                                        layout, type.getMemorySpace());
        Value view = builder.create<memref::ReinterpretCastOp>(
            alloc.getLoc(), viewType, workspace,
            builder.getIndexAttr(currentOffset), sizes, strides);
        alloc.replaceAllUsesWith(view);
        alloc.erase();
        currentOffset += elementCount;
      }
      function->setAttr("npu.workspace_elements",
                        builder.getI64IntegerAttr(currentOffset));
      function->setAttr(
          "npu.workspace_alignment",
          builder.getI64IntegerAttr(requiredWorkspaceAlignment));
      llvm::outs() << "[npu-plan-static-workspace] " << function.getName()
                   << ": " << currentOffset << " f32 elements, alignment "
                   << requiredWorkspaceAlignment << " bytes\n";
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass> npu::createPlanStaticWorkspacePass() {
  return std::make_unique<PlanStaticWorkspacePass>();
}
