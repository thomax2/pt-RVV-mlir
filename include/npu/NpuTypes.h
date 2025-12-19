#ifndef NPU_TYPES_H
#define NPU_TYPES_H

#include "npu/NpuDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

#include "llvm/Support/MathExtras.h"

#define GET_TYPEDEF_CLASSES
#include "npu/NpuTypes.h.inc"

#endif