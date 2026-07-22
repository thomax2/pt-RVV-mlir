add_mlir_dialect_library(EotDialect
  EotDialect.cpp
  EotOps.cpp
  EotShapeUtils.cpp
  DEPENDS MLIREotIncGen
  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
)
