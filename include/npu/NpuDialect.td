#ifndef NPU_DIALECT_TD
#define NPU_DIALECT_TD


include "mlir/IR/OpBase.td"

def NpuDialect : Dialect {
    let name = "npu";
    let cppNamespace = "::npu";
    let summary = "Npu Dialect";
    let useDefaultTypePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];
}

#endif
