#ifndef EOT_SHAPE_UTILS_H
#define EOT_SHAPE_UTILS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace eot {

bool areCompatibleDims(int64_t lhs, int64_t rhs);

mlir::FailureOr<mlir::RankedTensorType>
requireRankedTensor(mlir::Operation *op, mlir::Value value,
                    llvm::StringRef name, int64_t rank,
                    mlir::Type elementType);

mlir::LogicalResult requireCompatibleDim(mlir::Operation *op,
                                         llvm::StringRef lhsName,
                                         int64_t lhs,
                                         llvm::StringRef rhsName,
                                         int64_t rhs);

mlir::LogicalResult requireShape(mlir::Operation *op, mlir::Value value,
                                 llvm::StringRef name,
                                 llvm::ArrayRef<int64_t> expected,
                                 mlir::Type elementType);

} // namespace eot

#endif
