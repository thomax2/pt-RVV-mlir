#include "TorchEotCustomToTosa.h"

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"

#include <cmath>

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {

static StringRef canonicalEotName(StringRef name) {
  if (name.consume_front("eot::") || name.consume_front("eot.")) {
    name.consume_back(".default");
    return name;
  }
  return {};
}

static std::string jsonNumber(double value) {
  return llvm::formatv("{0}", llvm::json::Value(value)).str();
}

static FailureOr<double> constantFloat(Operation *op, Value value,
                                       StringRef label,
                                       ConversionPatternRewriter &rewriter) {
  double result;
  if (!matchPattern(value, m_TorchConstantFloat(&result)) ||
      !std::isfinite(result)) {
    (void)rewriter.notifyMatchFailure(
        op, llvm::formatv("{0} must be a finite compile-time float", label));
    return failure();
  }
  return result;
}

struct Spec {
  unsigned tensorInputs;
  unsigned scalarInputs;
  unsigned results;
};

static const llvm::StringMap<Spec> &getSpecs() {
  static const llvm::StringMap<Spec> specs = {
      {"point_features", {2, 2, 1}}, {"masked_mean", {2, 0, 1}},
      {"masked_variance", {2, 0, 1}}, {"masked_max", {2, 1, 1}},
      {"masked_min", {2, 1, 1}}, {"gru_cell", {6, 0, 1}},
      {"gm_parameterize", {5, 7, 9}},
  };
  return specs;
}

static FailureOr<std::string>
encodeAttrs(OperatorOp op, StringRef name,
            ConversionPatternRewriter &rewriter) {
  ValueRange operands = op.getOperands();
  auto scalar = [&](unsigned index, StringRef label) {
    return constantFloat(op, operands[index], label, rewriter);
  };
  if (name == "point_features") {
    auto radius = scalar(2, "radius_epsilon");
    auto theta = scalar(3, "theta_epsilon");
    if (failed(radius) || failed(theta))
      return failure();
    return llvm::formatv(
               "{{\"radius_epsilon\":{0},\"schema_version\":1,"
               "\"theta_epsilon\":{1}}}",
               jsonNumber(*radius), jsonNumber(*theta))
        .str();
  }
  if (name == "masked_max" || name == "masked_min") {
    auto empty = scalar(2, "empty_value");
    if (failed(empty))
      return failure();
    return llvm::formatv(
               "{{\"empty_value\":{0},\"schema_version\":1}}",
               jsonNumber(*empty))
        .str();
  }
  if (name == "gm_parameterize") {
    SmallVector<double> values;
    constexpr llvm::StringLiteral labels[] = {
        "c_min",       "c_max",      "tau_min",    "tau_max",
        "alpha_base", "logvar_min", "logvar_max"};
    for (auto [index, label] : llvm::enumerate(labels)) {
      auto value = scalar(5 + index, label);
      if (failed(value))
        return failure();
      values.push_back(*value);
    }
    return llvm::formatv(
               "{{\"alpha_base\":{0},\"c_max\":{1},\"c_min\":{2},"
               "\"logvar_max\":{3},\"logvar_min\":{4},"
               "\"schema_version\":1,\"tau_max\":{5},\"tau_min\":{6}}}",
               jsonNumber(values[4]), jsonNumber(values[1]),
               jsonNumber(values[0]), jsonNumber(values[6]),
               jsonNumber(values[5]), jsonNumber(values[3]),
               jsonNumber(values[2]))
        .str();
  }
  return std::string("{\"schema_version\":1}");
}

struct ConvertTorchEotOperator : OpConversionPattern<OperatorOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OperatorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef name = canonicalEotName(op.getName());
    if (name.empty())
      return failure();
    auto specIt = getSpecs().find(name);
    if (specIt == getSpecs().end())
      return rewriter.notifyMatchFailure(op, "unsupported eot custom operator");
    const Spec &spec = specIt->second;
    if (op.getNumOperands() != spec.tensorInputs + spec.scalarInputs ||
        op.getNumResults() != spec.results)
      return rewriter.notifyMatchFailure(op, "EOT operand/result count mismatch");

    SmallVector<Value> tensorOperands;
    for (Value value : adaptor.getOperands().take_front(spec.tensorInputs)) {
      if (!isa<RankedTensorType>(value.getType()))
        return rewriter.notifyMatchFailure(
            op, "EOT tensor operand was not converted to RankedTensorType");
      tensorOperands.push_back(value);
    }
    SmallVector<Type> resultTypes;
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(),
                                                resultTypes)) ||
        llvm::any_of(resultTypes,
                     [](Type type) { return !isa<RankedTensorType>(type); }))
      return rewriter.notifyMatchFailure(
          op, "EOT results must convert to ranked builtin tensors");
    FailureOr<std::string> implementationAttrs =
        encodeAttrs(op, name, rewriter);
    if (failed(implementationAttrs))
      return failure();

    OperationState state(op.getLoc(), tosa::CustomOp::getOperationName());
    state.addOperands(tensorOperands);
    state.addTypes(resultTypes);
    state.addAttribute("domain_name", rewriter.getStringAttr("eot"));
    state.addAttribute("operator_name", rewriter.getStringAttr(name));
    state.addAttribute("implementation_attrs",
                       rewriter.getStringAttr(*implementationAttrs));
    Operation *custom = rewriter.create(state);
    rewriter.replaceOp(op, custom->getResults());
    return success();
  }
};

} // namespace

void mlir::torch::populateTorchEotCustomToTosaPatterns(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target) {
  target.addDynamicallyLegalOp<OperatorOp>([](OperatorOp op) {
    return canonicalEotName(op.getName()).empty();
  });
  patterns.add<ConvertTorchEotOperator>(typeConverter, patterns.getContext());
}
