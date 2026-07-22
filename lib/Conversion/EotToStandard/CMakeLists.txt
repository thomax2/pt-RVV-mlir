add_mlir_conversion_library(EotToStandard
  ConvertEotToStandard.cpp
  DEPENDS MLIREotTransformsIncGen
  LINK_LIBS PUBLIC
  EotDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRTosaDialect
  MLIRDialectUtils
  MLIRTransforms
)
