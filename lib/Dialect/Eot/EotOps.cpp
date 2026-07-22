#include "eot/EotOps.h"
#include "eot/EotShapeUtils.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace eot;

static LogicalResult verifyMaskedReduction(Operation *op, Value input,
                                           Value mask, Value result,
                                           int64_t axis) {
  Type f32 = Float32Type::get(op->getContext());
  Type i1 = IntegerType::get(op->getContext(), 1);
  auto inputType = requireRankedTensor(op, input, "input", 3, f32);
  auto maskType = requireRankedTensor(op, mask, "mask", 2, i1);
  auto resultType = requireRankedTensor(op, result, "result", 2, f32);
  if (failed(inputType) || failed(maskType) || failed(resultType))
    return failure();
  if (axis != 1)
    return op->emitOpError() << "supports only axis = 1, got " << axis;
  if (failed(requireCompatibleDim(op, "input batch", (*inputType).getDimSize(0),
                                  "mask batch", (*maskType).getDimSize(0))) ||
      failed(requireCompatibleDim(op, "input point", (*inputType).getDimSize(1),
                                  "mask point", (*maskType).getDimSize(1))) ||
      failed(requireCompatibleDim(op, "input batch", (*inputType).getDimSize(0),
                                  "result batch", (*resultType).getDimSize(0))) ||
      failed(requireCompatibleDim(op, "input channel", (*inputType).getDimSize(2),
                                  "result channel", (*resultType).getDimSize(1))))
    return failure();
  return success();
}

LogicalResult PointFeaturesOp::verify() {
  Type f32 = Float32Type::get(getContext());
  Type i1 = IntegerType::get(getContext(), 1);
  auto points = requireRankedTensor(*this, getPoints(), "points", 3, f32);
  auto mask = requireRankedTensor(*this, getMask(), "mask", 2, i1);
  auto result = requireRankedTensor(*this, getFeatures(), "features", 3, f32);
  if (failed(points) || failed(mask) || failed(result))
    return failure();
  if (!areCompatibleDims((*points).getDimSize(2), 2))
    return emitOpError() << "expected points shape [B, N, 2], got " << *points;
  if (!areCompatibleDims((*result).getDimSize(2), 9))
    return emitOpError() << "expected features shape [B, N, 9], got " << *result;
  if (getRadiusEpsilonAttr().getValueAsDouble() <= 0.0 ||
      getThetaEpsilonAttr().getValueAsDouble() < 0.0)
    return emitOpError("requires radius_epsilon > 0 and theta_epsilon >= 0");
  for (int64_t dim = 0; dim < 2; ++dim) {
    if (failed(requireCompatibleDim(*this, "points", (*points).getDimSize(dim),
                                    "mask", (*mask).getDimSize(dim))) ||
        failed(requireCompatibleDim(*this, "points", (*points).getDimSize(dim),
                                    "features", (*result).getDimSize(dim))))
      return failure();
  }
  return success();
}

LogicalResult MaskedMeanOp::verify() {
  return verifyMaskedReduction(*this, getInput(), getMask(), getResult(),
                               getAxisAttr().getInt());
}

LogicalResult MaskedVarianceOp::verify() {
  return verifyMaskedReduction(*this, getInput(), getMask(), getResult(),
                               getAxisAttr().getInt());
}

LogicalResult MaskedMaxOp::verify() {
  return verifyMaskedReduction(*this, getInput(), getMask(), getResult(),
                               getAxisAttr().getInt());
}

LogicalResult MaskedMinOp::verify() {
  return verifyMaskedReduction(*this, getInput(), getMask(), getResult(),
                               getAxisAttr().getInt());
}

LogicalResult GruCellOp::verify() {
  Type f32 = Float32Type::get(getContext());
  auto x = requireRankedTensor(*this, getX(), "x", 2, f32);
  auto h = requireRankedTensor(*this, getHPrev(), "h_prev", 2, f32);
  auto wih = requireRankedTensor(*this, getWeightIh(), "weight_ih", 2, f32);
  auto whh = requireRankedTensor(*this, getWeightHh(), "weight_hh", 2, f32);
  auto bih = requireRankedTensor(*this, getBiasIh(), "bias_ih", 1, f32);
  auto bhh = requireRankedTensor(*this, getBiasHh(), "bias_hh", 1, f32);
  auto out = requireRankedTensor(*this, getHNext(), "h_next", 2, f32);
  if (failed(x) || failed(h) || failed(wih) || failed(whh) || failed(bih) ||
      failed(bhh) || failed(out))
    return failure();
  if (getGateOrderAttr().getValue() != "rzn" ||
      !getLinearBeforeResetAttr().getValue())
    return emitOpError("supports only gate_order = \"rzn\" and linear_before_reset = true");
  int64_t i = (*x).getDimSize(1);
  int64_t hidden = (*h).getDimSize(1);
  if (ShapedType::isDynamic(i) || ShapedType::isDynamic(hidden))
    return emitOpError("requires static input size I and hidden size H; batch may be dynamic");
  int64_t gates = 3 * hidden;
  if (failed(requireCompatibleDim(*this, "x batch", (*x).getDimSize(0),
                                  "h_prev batch", (*h).getDimSize(0))) ||
      failed(requireShape(*this, getWeightIh(), "weight_ih", {gates, i}, f32)) ||
      failed(requireShape(*this, getWeightHh(), "weight_hh", {gates, hidden}, f32)) ||
      failed(requireShape(*this, getBiasIh(), "bias_ih", {gates}, f32)) ||
      failed(requireShape(*this, getBiasHh(), "bias_hh", {gates}, f32)) ||
      failed(requireShape(*this, getHNext(), "h_next", {(*x).getDimSize(0), hidden}, f32)))
    return failure();
  return success();
}

LogicalResult GmParameterizeOp::verify() {
  Type f32 = Float32Type::get(getContext());
  SmallVector<std::pair<Value, int64_t>> inputs = {
      {getPiLogits(), 4}, {getTauLogits(), 1}, {getCXi(), 2},
      {getRawLogvar(), 2}, {getQN(), 1}};
  int64_t batch = ShapedType::kDynamic;
  for (auto [value, width] : inputs) {
    auto type = dyn_cast<RankedTensorType>(value.getType());
    if (!type || type.getRank() != 2 || type.getElementType() != f32)
      return emitOpError() << "requires every input to be a rank-2 f32 tensor, got "
                           << value.getType();
    if (!areCompatibleDims(type.getDimSize(1), width))
      return emitOpError() << "expected input trailing dimension " << width
                           << ", got " << type;
    if (ShapedType::isDynamic(batch))
      batch = type.getDimSize(0);
    else if (!areCompatibleDims(batch, type.getDimSize(0)))
      return emitOpError("requires matching input batch dimensions");
  }
  SmallVector<std::pair<Value, int64_t>> outputs = {
      {getPi(), 4},       {getTau(), 1},      {getAlphaPi(), 4},
      {getAlpha0(), 1},   {getSPi(), 1},      {getCMean(), 2},
      {getCLogvar(), 2},  {getCVar(), 2},     {getQC(), 2}};
  for (auto [value, width] : outputs)
    if (failed(requireShape(*this, value, "output", {batch, width}, f32)))
      return failure();
  double cMin = getCMinAttr().getValueAsDouble();
  double cMax = getCMaxAttr().getValueAsDouble();
  double tauMin = getTauMinAttr().getValueAsDouble();
  double tauMax = getTauMaxAttr().getValueAsDouble();
  double lvMin = getLogvarMinAttr().getValueAsDouble();
  double lvMax = getLogvarMaxAttr().getValueAsDouble();
  if (!(cMin < cMax) || tauMin < 0.0 || tauMax <= 0.0 ||
      !(lvMin < lvMax))
    return emitOpError("requires c_min < c_max, tau_min >= 0, tau_max > 0, and logvar_min < logvar_max");
  return success();
}
