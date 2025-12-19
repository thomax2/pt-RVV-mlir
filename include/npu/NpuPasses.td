#ifndef NPU_PASSES_TD
#define NPU_PASSES_TD

include "mlir/Pass/PassBase.td"

def ConvertTosaToNpu : Pass<"convert-tosa-to-npu"> {
    let summary = "Convert Tosa To Npu";
    let constructor = "npu::createConvertTosaToNpuPass()";
    let options = [
        Option<"name", "name", "std::string", "", "help">
    ];
}

def ConvertNpuToVecScf : Pass<"convert-npu-to-vecscf"> {
    let summary = "Convert Npu To VecScf";
    let constructor = "npu::createConvertNpuToVecScfPass()";
    let options = [
        Option<"name", "name", "std::string", "", "help">
    ];
}

#endif
