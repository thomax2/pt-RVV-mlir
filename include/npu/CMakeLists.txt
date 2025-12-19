add_mlir_dialect(Npu npu)

set(LLVM_TARGET_DEFINITIONS NpuPasses.td)
mlir_tablegen(NpuPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIRNpuTransformsIncGen)

add_custom_target(header DEPENDS MLIRNpuIncGen MLIRNpuTransformsIncGen)