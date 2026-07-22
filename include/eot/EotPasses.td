#ifndef EOT_PASSES_TD
#define EOT_PASSES_TD
include "mlir/Pass/PassBase.td"

def ConvertTosaCustomToEot : Pass<"convert-tosa-custom-to-eot", "mlir::ModuleOp"> {
  let summary = "Convert domain=eot TOSA custom operations to EOT operations";
  let constructor = "eot::createConvertTosaCustomToEotPass()";
  let options = [Option<"strict", "strict", "bool", "false",
    "Reject non-EOT tosa.custom operations">];
}

def ConvertEotToStandard : Pass<"convert-eot-to-standard", "mlir::ModuleOp"> {
  let summary = "Lower EOT domain operations to tensor/linalg/arith/math";
  let constructor = "eot::createConvertEotToStandardPass()";
}
#endif
