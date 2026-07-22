#ifndef EOT_DIALECT_TD
#define EOT_DIALECT_TD

include "mlir/IR/OpBase.td"

def EotDialect : Dialect {
  let name = "eot";
  let cppNamespace = "::eot";
  let summary = "Extended-object-tracking model domain operations";
}

#endif
