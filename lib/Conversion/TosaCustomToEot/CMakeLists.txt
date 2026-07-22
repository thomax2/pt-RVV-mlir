add_mlir_conversion_library(TosaCustomToEot
  ConvertTosaCustomToEot.cpp
  DEPENDS MLIREotTransformsIncGen
  LINK_LIBS PUBLIC
  EotDialect
  MLIRIR
  MLIRTosaDialect
  MLIRTransforms
  LLVMSupport
)
