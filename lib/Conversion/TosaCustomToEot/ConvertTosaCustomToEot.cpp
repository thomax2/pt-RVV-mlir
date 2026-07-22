#include "eot/EotOps.h"
#include "eot/EotPasses.h"

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/JSON.h"

#include <cmath>
#include <initializer_list>

namespace eot {
#define GEN_PASS_DEF_CONVERTTOSACUSTOMTOEOT
#include "eot/EotPasses.h.inc"
} // namespace eot

using namespace mlir;

namespace {

static FailureOr<const llvm::json::Object *>
parseAttrs(tosa::CustomOp op, llvm::json::Value &storage) {
  auto text = op->getAttrOfType<StringAttr>("implementation_attrs");
  if (!text) {
    op.emitOpError("requires string implementation_attrs");
    return failure();
  }
  auto parsed = llvm::json::parse(text.getValue());
  if (!parsed) {
    op.emitOpError("invalid implementation_attrs JSON: ")
        << llvm::toString(parsed.takeError());
    return failure();
  }
  storage = std::move(*parsed);
  auto *object = storage.getAsObject();
  if (!object) {
    op.emitOpError("implementation_attrs must be a JSON object");
    return failure();
  }
  auto schema = object->getInteger("schema_version");
  if (!schema || *schema != 1) {
    op.emitOpError("requires schema_version = 1");
    return failure();
  }
  return object;
}

static FailureOr<double> getNumber(tosa::CustomOp op,
                                   const llvm::json::Object &object,
                                   StringRef key, double defaultValue) {
  const llvm::json::Value *value = object.get(key);
  if (!value)
    return defaultValue;
  if (auto number = value->getAsNumber()) {
    if (std::isfinite(*number))
      return *number;
    op.emitOpError() << "JSON field '" << key << "' must be finite";
    return failure();
  }
  op.emitOpError() << "JSON field '" << key << "' must be numeric";
  return failure();
}

static LogicalResult rejectUnknownFields(tosa::CustomOp op,
                                         const llvm::json::Object &object,
                                         std::initializer_list<StringRef> allowed) {
  llvm::StringSet<> names;
  names.insert("schema_version");
  for (StringRef name : allowed)
    names.insert(name);
  for (const auto &entry : object)
    if (!names.count(entry.first))
      return op.emitOpError() << "unknown implementation_attrs field '"
                              << entry.first.str() << "'";
  return success();
}

struct ConvertEotCustomPattern : OpConversionPattern<tosa::CustomOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tosa::CustomOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto domain = op->getAttrOfType<StringAttr>("domain_name");
    auto name = op->getAttrOfType<StringAttr>("operator_name");
    if (!domain || domain.getValue() != "eot")
      return failure();
    if (!name)
      return op.emitOpError("EOT custom operation requires operator_name");

    struct Spec {
      llvm::StringLiteral mlirName;
      unsigned inputs;
      unsigned results;
    };
    static const llvm::StringMap<Spec> specs = {
        {"point_features", {"eot.point_features", 2, 1}},
        {"masked_mean", {"eot.masked_mean", 2, 1}},
        {"masked_variance", {"eot.masked_variance", 2, 1}},
        {"masked_max", {"eot.masked_max", 2, 1}},
        {"masked_min", {"eot.masked_min", 2, 1}},
        {"gru_cell", {"eot.gru_cell", 6, 1}},
        {"gm_parameterize", {"eot.gm_parameterize", 5, 9}},
    };
    auto it = specs.find(name.getValue());
    if (it == specs.end())
      return op.emitOpError() << "unsupported EOT operator '" << name.getValue()
                              << "'";
    const Spec &spec = it->second;
    if (adaptor.getInputList().size() != spec.inputs ||
        op->getNumResults() != spec.results)
      return op.emitOpError() << "operator '" << name.getValue() << "' expects "
                              << spec.inputs << " tensor inputs and "
                              << spec.results << " results";
    for (Type type : adaptor.getInputList().getTypes())
      if (!isa<RankedTensorType>(type))
        return op.emitOpError("EOT custom inputs must be ranked tensors");

    llvm::json::Value json(nullptr);
    FailureOr<const llvm::json::Object *> object = parseAttrs(op, json);
    if (failed(object))
      return failure();

    OperationState state(op.getLoc(), spec.mlirName);
    state.addOperands(adaptor.getInputList());
    state.addTypes(op->getResultTypes());
    auto addF64 = [&](StringRef key, double defaultValue) -> LogicalResult {
      FailureOr<double> value = getNumber(op, **object, key, defaultValue);
      if (failed(value))
        return failure();
      state.addAttribute(key, rewriter.getF64FloatAttr(*value));
      return success();
    };
    if (name.getValue() == "point_features") {
      if (failed(rejectUnknownFields(op, **object,
                                     {"radius_epsilon", "theta_epsilon"})))
        return failure();
      if (failed(addF64("radius_epsilon", 1e-8)) ||
          failed(addF64("theta_epsilon", 1e-10)))
        return failure();
    } else if (name.getValue() == "masked_mean" ||
               name.getValue() == "masked_variance") {
      if (failed(rejectUnknownFields(op, **object, {})))
        return failure();
      state.addAttribute("axis", rewriter.getI64IntegerAttr(1));
    } else if (name.getValue() == "masked_max" ||
               name.getValue() == "masked_min") {
      if (failed(rejectUnknownFields(op, **object, {"empty_value"})))
        return failure();
      state.addAttribute("axis", rewriter.getI64IntegerAttr(1));
      double defaultValue = name.getValue() == "masked_max" ? -1e9 : 1e9;
      if (failed(addF64("empty_value", defaultValue)))
        return failure();
    } else if (name.getValue() == "gru_cell") {
      if (failed(rejectUnknownFields(op, **object, {})))
        return failure();
      state.addAttribute("gate_order", rewriter.getStringAttr("rzn"));
      state.addAttribute("linear_before_reset", rewriter.getBoolAttr(true));
    } else {
      if (failed(rejectUnknownFields(
              op, **object,
              {"c_min", "c_max", "tau_min", "tau_max", "alpha_base",
               "logvar_min", "logvar_max"})))
        return failure();
      static constexpr std::pair<llvm::StringLiteral, double> defaults[] = {
          {"c_min", 0.003},       {"c_max", 0.70},
          {"tau_min", 1.0},       {"tau_max", 100.0},
          {"alpha_base", 1.0},    {"logvar_min", -6.0},
          {"logvar_max", 1.5}};
      for (auto [key, value] : defaults)
        if (failed(addF64(key, value)))
          return failure();
    }
    Operation *replacement = rewriter.create(state);
    rewriter.replaceOp(op, replacement->getResults());
    return success();
  }
};

struct ConvertTosaCustomToEotPass
    : ::eot::impl::ConvertTosaCustomToEotBase<ConvertTosaCustomToEotPass> {
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<eot::EotDialect, tosa::TosaDialect>();
  }
  void runOnOperation() final {
    ConversionTarget target(getContext());
    target.addLegalDialect<eot::EotDialect, tosa::TosaDialect,
                           BuiltinDialect>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    target.addDynamicallyLegalOp<tosa::CustomOp>([&](tosa::CustomOp op) {
      auto domain = op->getAttrOfType<StringAttr>("domain_name");
      if (domain && domain.getValue() == "eot")
        return false;
      return !strict;
    });
    RewritePatternSet patterns(&getContext());
    patterns.add<ConvertEotCustomPattern>(&getContext());
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<mlir::Pass> eot::createConvertTosaCustomToEotPass() {
  return std::make_unique<ConvertTosaCustomToEotPass>();
}
