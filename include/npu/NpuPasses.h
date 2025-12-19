#ifndef NPU_PASS_H
#define NPU_PASS_H

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "npu/NpuOps.h"
#include <memory>

namespace npu {

// 先生成定义
#define GEN_PASS_DECL
#include "npu/NpuPasses.h.inc"

// 在写 create 函数表
std::unique_ptr<mlir::Pass> createConvertTosaToNpuPass(ConvertTosaToNpuOptions options={});
std::unique_ptr<mlir::Pass> createConvertNpuToVecScfPass(ConvertNpuToVecScfOptions options={});


// 生成注册函数
#define GEN_PASS_REGISTRATION
#include "npu/NpuPasses.h.inc"
}

#endif