#include "eot/EotDialect.h"
#include "eot/EotOps.h"

using namespace mlir;

#include "eot/EotDialect.cpp.inc"

#define GET_OP_CLASSES
#include "eot/Eot.cpp.inc"

void eot::EotDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "eot/Eot.cpp.inc"
      >();
}
