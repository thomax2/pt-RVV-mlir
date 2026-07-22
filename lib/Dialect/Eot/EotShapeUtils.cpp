#include "eot/EotShapeUtils.h"

using namespace mlir;

bool eot::areCompatibleDims(int64_t lhs, int64_t rhs) {
  return ShapedType::isDynamic(lhs) || ShapedType::isDynamic(rhs) || lhs == rhs;
}

FailureOr<RankedTensorType>
eot::requireRankedTensor(Operation *op, Value value, StringRef name,
                         int64_t rank, Type elementType) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type) {
    op->emitOpError() << "expected '" << name
                      << "' to be a ranked tensor, got " << value.getType();
    return failure();
  }
  if (type.getRank() != rank) {
    op->emitOpError() << "expected '" << name << "' to have rank " << rank
                      << ", got " << type;
    return failure();
  }
  if (type.getElementType() != elementType) {
    op->emitOpError() << "expected '" << name << "' element type "
                      << elementType << ", got " << type.getElementType();
    return failure();
  }
  return type;
}

LogicalResult eot::requireCompatibleDim(Operation *op, StringRef lhsName,
                                        int64_t lhs, StringRef rhsName,
                                        int64_t rhs) {
  if (areCompatibleDims(lhs, rhs))
    return success();
  return op->emitOpError() << "incompatible " << lhsName << " dimension "
                           << lhs << " and " << rhsName << " dimension " << rhs;
}

LogicalResult eot::requireShape(Operation *op, Value value, StringRef name,
                                ArrayRef<int64_t> expected, Type elementType) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type)
    return op->emitOpError() << "expected '" << name
                             << "' to be a ranked tensor, got "
                             << value.getType();
  if (type.getRank() != static_cast<int64_t>(expected.size()))
    return op->emitOpError() << "expected '" << name << "' rank "
                             << expected.size() << ", got " << type;
  if (type.getElementType() != elementType)
    return op->emitOpError() << "expected '" << name << "' element type "
                             << elementType << ", got "
                             << type.getElementType();
  for (int64_t index = 0; index < type.getRank(); ++index) {
    int64_t actual = type.getDimSize(index);
    int64_t wanted = expected[index];
    if (!areCompatibleDims(actual, wanted))
      return op->emitOpError() << "expected '" << name << "' dimension "
                               << index << " to be " << wanted << ", got "
                               << actual;
  }
  return success();
}
