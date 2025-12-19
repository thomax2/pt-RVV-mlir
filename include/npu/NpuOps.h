#ifndef NPU_OPS_H
#define NPU_OPS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "npu/NpuDialect.h"
#include "npu/NpuTypes.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/IR/RegionKindInterface.h"

#define GET_OP_CLASSES
#include "npu/Npu.h.inc"

#endif
