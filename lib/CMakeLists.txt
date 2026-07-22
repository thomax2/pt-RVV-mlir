add_mlir_library(Npu
  npu.cpp
  DEPENDS MLIRNpuIncGen
  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
)
add_subdirectory(Transforms)
add_subdirectory(Dialect/Eot)
add_subdirectory(Conversion/EotToStandard)
add_subdirectory(Conversion/TosaCustomToEot)
