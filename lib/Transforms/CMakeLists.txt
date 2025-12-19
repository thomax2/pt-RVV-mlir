add_mlir_library(
  NpuTransforms
  ConvertTosaToNpu.cpp
  ConvertNpuToVecScf.cpp
  DEPENDS MLIRNpuTransformsIncGen
  
  LINK_LIBS MLIRTosaDialect MLIRSCFDialect MLIRVectorDialect MLIRArithDialect
  # MLIRTosaToLLVM
)