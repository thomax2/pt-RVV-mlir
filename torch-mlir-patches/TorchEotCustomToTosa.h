#ifndef TORCH_MLIR_CONVERSION_TORCH_TO_TOSA_TORCHEOTCUSTOMTOTOSA_H
#define TORCH_MLIR_CONVERSION_TORCH_TO_TOSA_TORCHEOTCUSTOMTOTOSA_H

#include "mlir/Transforms/DialectConversion.h"

namespace mlir::torch {

/// Adds EOT torch.operator legality and rewrite rules to the existing
/// Torch-to-TOSA conversion. The caller must pass that conversion's own
/// TypeConverter so vtensor operands/results use its normal materializations.
void populateTorchEotCustomToTosaPatterns(TypeConverter &typeConverter,
                                          RewritePatternSet &patterns,
                                          ConversionTarget &target);

} // namespace mlir::torch

#endif
