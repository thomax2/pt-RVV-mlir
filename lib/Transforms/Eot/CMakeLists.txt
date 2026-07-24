add_mlir_library(
  EotTransforms
  PlanStaticWorkspace.cpp
  DEPENDS MLIREotTransformsIncGen

  LINK_LIBS
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRPass
  MLIRSupport
)
