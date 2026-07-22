add_mlir_library(
  NpuTransforms
  ConvertTosaToNpu.cpp
  ConvertNpuToVecScf.cpp
  PlanStaticWorkspace.cpp
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
