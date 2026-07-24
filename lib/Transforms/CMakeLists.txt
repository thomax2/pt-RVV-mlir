add_mlir_library(
  NpuTransforms
  ConvertTosaToNpu.cpp
  ConvertNpuToVecScf.cpp
  DEPENDS MLIRNpuTransformsIncGen
  
  LINK_LIBS
  Npu
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRTosaDialect
  MLIRDialectUtils
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  # MLIRTosaToLLVM
)

add_subdirectory(Eot)
