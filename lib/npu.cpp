#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "npu/NpuDialect.h"
#include "npu/NpuOps.h"


#include "npu/NpuTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace npu;

static ParseResult parseShape(AsmParser &parser, 
  llvm::SmallVectorImpl<int64_t> &shape) {
  // parseDimensionList 是 MLIR 内置的，用于解析 'x' 分隔的维度
  return parser.parseDimensionList(shape, true, true);
}

// 打印函数：输出 1x2x3
static void printShape(AsmPrinter &printer, llvm::ArrayRef<int64_t> shape) {
  for (int64_t dim : shape) {
    // 简单粗暴：每个维度后面都跟一个 x
    // 输出效果：1x8x8x1x
    printer << dim << "x";
  }
}


#define GET_TYPEDEF_CLASSES
#include "npu/NpuTypes.cpp.inc"

#include "npu/NpuDialect.cpp.inc"

#define GET_OP_CLASSES
#include "npu/Npu.cpp.inc"


void NpuDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "npu/Npu.cpp.inc"
  >();
  registerTypes();
}

void NpuDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "npu/NpuTypes.cpp.inc"
      >();
}
