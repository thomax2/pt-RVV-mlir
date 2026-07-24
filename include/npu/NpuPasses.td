#ifndef NPU_PASSES_TD
#define NPU_PASSES_TD

include "mlir/Pass/PassBase.td"

def ConvertTosaToNpu : Pass<"convert-tosa-to-npu"> {
    let summary = "Legacy: convert generic TOSA operations to legacy NPU operations";
    let constructor = "npu::createConvertTosaToNpuPass()";
    let options = [
        Option<"name", "name", "std::string", "", "help">
    ];
}

def ConvertNpuToVecScf : Pass<"convert-npu-to-vecscf"> {
    let summary = "Legacy: lower legacy NPU operations to scalar SCF and MemRef";
    let constructor = "npu::createConvertNpuToVecScfPass()";
    let options = [
        Option<"name", "name", "std::string", "", "help">
    ];
}

def PlanStaticWorkspace : Pass<"npu-plan-static-workspace", "mlir::ModuleOp"> {
    let summary = "Plan static memref allocations in an explicit byte workspace";
    let constructor = "npu::createPlanStaticWorkspacePass()";
}

#endif
