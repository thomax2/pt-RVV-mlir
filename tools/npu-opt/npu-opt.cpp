#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
// 导入 Func Dialect
#include "mlir/Dialect/Func/IR/FuncOps.h"
// 导入 MLIR 自带 Pass
#include "mlir/Transforms/Passes.h"
// 导入我们新建的 Dialect
#include "npu/NpuDialect.h"
#include "npu/NpuPasses.h"
#include "eot/EotDialect.h"
#include "eot/EotPasses.h"
using namespace mlir;
using namespace llvm;

int main(int argc, char ** argv) {
  DialectRegistry registry;
  registerAllDialects(registry);
  registry.insert<npu::NpuDialect, eot::EotDialect>();
  registerAllPasses();
  npu::registerPasses();
  eot::registerPasses();
  return asMainReturnCode(MlirOptMain(argc, argv, "npu-opt", registry));
}
