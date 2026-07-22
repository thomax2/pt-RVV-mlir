#include "npu/NpuPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#define GEN_PASS_DEF_PLANSTATICWORKSPACE
#include "npu/NpuPasses.h.inc"

using namespace mlir;

namespace {
// Despite the compatibility command name, this pass operates only on Func and
// MemRef IR and has no dependency on the legacy NPU dialect.
struct PlanStaticWorkspacePass
    : npu::impl::PlanStaticWorkspaceBase<PlanStaticWorkspacePass> {
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<func::FuncDialect, memref::MemRefDialect>();
  }

  void runOnOperation() final {
    OpBuilder builder(&getContext());
    for (func::FuncOp function : getOperation().getOps<func::FuncOp>()) {
      if (function.isExternal())
        continue;
      SmallVector<memref::AllocOp> allocations;
      function.walk([&](memref::AllocOp alloc) { allocations.push_back(alloc); });
      if (allocations.empty())
        continue;
      for (memref::AllocOp alloc : allocations) {
        if (!alloc.getType().hasStaticShape() ||
            !alloc.getType().getElementType().isF32()) {
          alloc.emitOpError("static workspace requires a static f32 memref");
          signalPassFailure();
          return;
        }
      }

      FunctionType oldType = function.getFunctionType();
      SmallVector<Type> oldInputs(oldType.getInputs());
      SmallVector<Type> oldResults(oldType.getResults());
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
        SmallVector<int64_t> staticStrides(type.getRank());
        int64_t elementCount = 1;
        for (int64_t dim : type.getShape())
          elementCount *= dim;
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
        currentOffset = llvm::alignTo(currentOffset + elementCount, int64_t{4});
      }
      function->setAttr("npu.workspace_elements",
                        builder.getI64IntegerAttr(currentOffset));
      llvm::outs() << "[npu-plan-static-workspace] " << function.getName()
                   << ": " << currentOffset << " f32 elements\n";
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass> npu::createPlanStaticWorkspacePass() {
  return std::make_unique<PlanStaticWorkspacePass>();
}
