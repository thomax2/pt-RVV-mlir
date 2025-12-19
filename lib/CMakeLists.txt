add_mlir_library(Npu npu.cpp DEPENDS MLIRNpuIncGen)
add_subdirectory(Transforms)