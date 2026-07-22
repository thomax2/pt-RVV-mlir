add_mlir_dialect(Eot eot)

set(LLVM_TARGET_DEFINITIONS EotPasses.td)
mlir_tablegen(EotPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIREotTransformsIncGen)

add_custom_target(eot-headers DEPENDS MLIREotIncGen MLIREotTransformsIncGen)
