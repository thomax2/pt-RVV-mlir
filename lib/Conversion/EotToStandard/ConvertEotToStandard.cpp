#include "eot/EotPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"

#include <cmath>
#include <limits>

#define GEN_PASS_DEF_CONVERTEOTTOSTANDARD
#include "eot/EotPasses.h.inc"

using namespace mlir;

namespace {

static Value constantF32(OpBuilder &builder, Location loc, double value) {
  return builder.create<arith::ConstantOp>(loc,
                                            builder.getF32FloatAttr(value));
}

static Value createEmpty(OpBuilder &builder, Location loc,
                         RankedTensorType type, ValueRange dynamicSizes = {}) {
  return builder.create<tensor::EmptyOp>(loc, type.getShape(),
                                         type.getElementType(), dynamicSizes);
}

static Value createFilled(OpBuilder &builder, Location loc,
                          RankedTensorType type, Value fill,
                          ValueRange dynamicSizes = {}) {
  Value empty = createEmpty(builder, loc, type, dynamicSizes);
  return builder.create<linalg::FillOp>(loc, fill, empty).getResult(0);
}

static SmallVector<Value> getDynamicDims(OpBuilder &builder, Location loc,
                                         Value shaped,
                                         ArrayRef<int64_t> sourceDims,
                                         RankedTensorType resultType) {
  SmallVector<Value> values;
  unsigned sourceIndex = 0;
  for (int64_t dim : resultType.getShape()) {
    int64_t sourceDim = sourceDims[sourceIndex++];
    if (ShapedType::isDynamic(dim)) {
      if (sourceDim >= 0)
        values.push_back(builder.create<tensor::DimOp>(loc, shaped, sourceDim));
      else
        values.push_back(
            builder.create<arith::ConstantIndexOp>(loc, -sourceDim));
    }
  }
  return values;
}

template <typename Body>
static Value createGeneric(OpBuilder &builder, Location loc, ValueRange inputs,
                           Value output, ArrayRef<AffineMap> maps,
                           ArrayRef<utils::IteratorType> iterators, Body body) {
  auto op = builder.create<linalg::GenericOp>(
      loc, TypeRange{output.getType()}, inputs, ValueRange{output}, maps,
      iterators,
      [&](OpBuilder &nested, Location nestedLoc, ValueRange args) {
        body(nested, nestedLoc, args);
      });
  return op.getResult(0);
}

struct PointFeaturesLowering
    : OpConversionPattern<eot::PointFeaturesOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(eot::PointFeaturesOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getFeatures().getType());
    SmallVector<Value> dynamicDims =
        getDynamicDims(rewriter, loc, adaptor.getPoints(), {0, 1, -9}, resultType);
    Value output = createEmpty(rewriter, loc, resultType, dynamicDims);
    MLIRContext *ctx = rewriter.getContext();
    AffineExpr b, n, k;
    bindDims(ctx, b, n, k);
    SmallVector<AffineMap> maps = {
        AffineMap::get(3, 0, {b, n}, ctx),
        AffineMap::get(3, 0, {b, n, k}, ctx)};
    SmallVector<utils::IteratorType> iterators(3,
                                               utils::IteratorType::parallel);
    Value result = createGeneric(
        rewriter, loc,
        ValueRange{adaptor.getMask()}, output, maps, iterators,
        [&](OpBuilder &bld, Location bodyLoc, ValueRange args) {
          Value batchIndex = bld.create<linalg::IndexOp>(bodyLoc, 0);
          Value pointIndex = bld.create<linalg::IndexOp>(bodyLoc, 1);
          Value xIndex = bld.create<arith::ConstantIndexOp>(bodyLoc, 0);
          Value yIndex = bld.create<arith::ConstantIndexOp>(bodyLoc, 1);
          Value x = bld.create<tensor::ExtractOp>(
              bodyLoc, adaptor.getPoints(),
              ValueRange{batchIndex, pointIndex, xIndex});
          Value y = bld.create<tensor::ExtractOp>(
              bodyLoc, adaptor.getPoints(),
              ValueRange{batchIndex, pointIndex, yIndex});
          Value mask = args[0];
          Value zero = constantF32(bld, bodyLoc, 0.0);
          Value one = constantF32(bld, bodyLoc, 1.0);
          Value xx = bld.create<arith::MulFOp>(bodyLoc, x, x);
          Value yy = bld.create<arith::MulFOp>(bodyLoc, y, y);
          Value xy = bld.create<arith::MulFOp>(bodyLoc, x, y);
          Value r2 = bld.create<arith::AddFOp>(bodyLoc, xx, yy);
          Value radiusEps = constantF32(
              bld, bodyLoc, op.getRadiusEpsilonAttr().getValueAsDouble());
          Value thetaEps = constantF32(
              bld, bodyLoc, op.getThetaEpsilonAttr().getValueAsDouble());
          Value r = bld.create<math::SqrtOp>(
              bodyLoc, bld.create<arith::AddFOp>(bodyLoc, r2, radiusEps));
          Value nonzero = bld.create<arith::CmpFOp>(
              bodyLoc, arith::CmpFPredicate::OGT, r2, thetaEps);
          Value thetaY = bld.create<arith::SelectOp>(bodyLoc, nonzero, y, zero);
          Value thetaX = bld.create<arith::SelectOp>(bodyLoc, nonzero, x, one);
          Value theta = bld.create<math::Atan2Op>(bodyLoc, thetaY, thetaX);
          SmallVector<Value> features = {
              x, y, bld.create<math::AbsFOp>(bodyLoc, x),
              bld.create<math::AbsFOp>(bodyLoc, y), r, theta, xx, yy, xy};
          Value index = bld.create<linalg::IndexOp>(bodyLoc, 2);
          Value selected = features.back();
          for (int64_t i = 7; i >= 0; --i) {
            Value wanted = bld.create<arith::ConstantIndexOp>(bodyLoc, i);
            Value isIndex = bld.create<arith::CmpIOp>(
                bodyLoc, arith::CmpIPredicate::eq, index, wanted);
            selected = bld.create<arith::SelectOp>(bodyLoc, isIndex,
                                                   features[i], selected);
          }
          selected =
              bld.create<arith::SelectOp>(bodyLoc, mask, selected, zero);
          bld.create<linalg::YieldOp>(bodyLoc, selected);
        });
    rewriter.replaceOp(op, result);
    return success();
  }
};

static Value lowerMaskedSum(ConversionPatternRewriter &rewriter, Location loc,
                            Value input, Value mask, RankedTensorType resultType,
                            ValueRange dynamicDims, Value mean = {}) {
  Value zero = constantF32(rewriter, loc, 0.0);
  Value init = createFilled(rewriter, loc, resultType, zero, dynamicDims);
  MLIRContext *ctx = rewriter.getContext();
  AffineExpr b, n, c;
  bindDims(ctx, b, n, c);
  SmallVector<AffineMap> maps = {
      AffineMap::get(3, 0, {b, n, c}, ctx),
      AffineMap::get(3, 0, {b, n}, ctx)};
  SmallVector<Value> inputs = {input, mask};
  if (mean) {
    inputs.push_back(mean);
    maps.push_back(AffineMap::get(3, 0, {b, c}, ctx));
  }
  maps.push_back(AffineMap::get(3, 0, {b, c}, ctx));
  SmallVector<utils::IteratorType> iterators = {
      utils::IteratorType::parallel, utils::IteratorType::reduction,
      utils::IteratorType::parallel};
  return createGeneric(
      rewriter, loc, inputs, init, maps, iterators,
      [&](OpBuilder &bld, Location bodyLoc, ValueRange args) {
        Value sample = args[0];
        unsigned accIndex = 2;
        if (mean) {
          Value delta = bld.create<arith::SubFOp>(bodyLoc, sample, args[2]);
          sample = bld.create<arith::MulFOp>(bodyLoc, delta, delta);
          accIndex = 3;
        }
        Value contribution = bld.create<arith::SelectOp>(
            bodyLoc, args[1], sample, constantF32(bld, bodyLoc, 0.0));
        Value sum =
            bld.create<arith::AddFOp>(bodyLoc, args[accIndex], contribution);
        bld.create<linalg::YieldOp>(bodyLoc, sum);
      });
}

static Value lowerMaskedCount(ConversionPatternRewriter &rewriter, Location loc,
                              Value input, Value mask,
                              RankedTensorType resultType,
                              ValueRange dynamicDims) {
  Value zero = constantF32(rewriter, loc, 0.0);
  Value init = createFilled(rewriter, loc, resultType, zero, dynamicDims);
  MLIRContext *ctx = rewriter.getContext();
  AffineExpr b, n, c;
  bindDims(ctx, b, n, c);
  SmallVector<AffineMap> maps = {
      AffineMap::get(3, 0, {b, n, c}, ctx),
      AffineMap::get(3, 0, {b, n}, ctx),
      AffineMap::get(3, 0, {b, c}, ctx)};
  SmallVector<utils::IteratorType> iterators = {
      utils::IteratorType::parallel, utils::IteratorType::reduction,
      utils::IteratorType::parallel};
  return createGeneric(
      rewriter, loc, ValueRange{input, mask}, init, maps, iterators,
      [&](OpBuilder &bld, Location bodyLoc, ValueRange args) {
        Value increment = bld.create<arith::SelectOp>(
            bodyLoc, args[1], constantF32(bld, bodyLoc, 1.0),
            constantF32(bld, bodyLoc, 0.0));
        Value count = bld.create<arith::AddFOp>(bodyLoc, args[2], increment);
        bld.create<linalg::YieldOp>(bodyLoc, count);
      });
}

static Value divideByCount(ConversionPatternRewriter &rewriter, Location loc,
                           Value sum, Value count,
                           RankedTensorType resultType,
                           ValueRange dynamicDims) {
  Value output = createEmpty(rewriter, loc, resultType, dynamicDims);
  AffineMap identity = rewriter.getMultiDimIdentityMap(2);
  SmallVector<AffineMap> maps = {identity, identity, identity};
  SmallVector<utils::IteratorType> iterators(2,
                                             utils::IteratorType::parallel);
  return createGeneric(
      rewriter, loc, ValueRange{sum, count}, output, maps, iterators,
      [&](OpBuilder &bld, Location bodyLoc, ValueRange args) {
        Value one = constantF32(bld, bodyLoc, 1.0);
        Value denom = bld.create<arith::MaximumFOp>(bodyLoc, args[1], one);
        bld.create<linalg::YieldOp>(
            bodyLoc, bld.create<arith::DivFOp>(bodyLoc, args[0], denom));
      });
}

template <typename OpTy, bool IsVariance>
struct MaskedAverageLowering : OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    SmallVector<Value> dims =
        getDynamicDims(rewriter, loc, adaptor.getInput(), {0, 2}, resultType);
    Value count = lowerMaskedCount(rewriter, loc, adaptor.getInput(),
                                   adaptor.getMask(), resultType, dims);
    Value sum = lowerMaskedSum(rewriter, loc, adaptor.getInput(),
                               adaptor.getMask(), resultType, dims);
    Value mean = divideByCount(rewriter, loc, sum, count, resultType, dims);
    if constexpr (IsVariance) {
      sum = lowerMaskedSum(rewriter, loc, adaptor.getInput(), adaptor.getMask(),
                           resultType, dims, mean);
      mean = divideByCount(rewriter, loc, sum, count, resultType, dims);
    }
    rewriter.replaceOp(op, mean);
    return success();
  }
};

template <typename OpTy, bool IsMax>
struct MaskedExtremaLowering : OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    SmallVector<Value> dims =
        getDynamicDims(rewriter, loc, adaptor.getInput(), {0, 2}, resultType);
    Value emptyValue = constantF32(
        rewriter, loc, op.getEmptyValueAttr().getValueAsDouble());
    Value init = createFilled(rewriter, loc, resultType, emptyValue, dims);
    MLIRContext *ctx = rewriter.getContext();
    AffineExpr b, n, c;
    bindDims(ctx, b, n, c);
    SmallVector<AffineMap> maps = {
        AffineMap::get(3, 0, {b, n, c}, ctx),
        AffineMap::get(3, 0, {b, n}, ctx),
        AffineMap::get(3, 0, {b, c}, ctx)};
    SmallVector<utils::IteratorType> iterators = {
        utils::IteratorType::parallel, utils::IteratorType::reduction,
        utils::IteratorType::parallel};
    Value result = createGeneric(
        rewriter, loc, ValueRange{adaptor.getInput(), adaptor.getMask()}, init,
        maps, iterators,
        [&](OpBuilder &bld, Location bodyLoc, ValueRange args) {
          Value sample = bld.create<arith::SelectOp>(bodyLoc, args[1], args[0],
                                                      emptyValue);
          Value value;
          if constexpr (IsMax)
            value = bld.create<arith::MaximumFOp>(bodyLoc, args[2], sample);
          else
            value = bld.create<arith::MinimumFOp>(bodyLoc, args[2], sample);
          bld.create<linalg::YieldOp>(bodyLoc, value);
        });
    rewriter.replaceOp(op, result);
    return success();
  }
};

static Value lowerMatmulProjection(ConversionPatternRewriter &rewriter,
                                   Location loc, Value lhs, Value rhs,
                                   int64_t outputWidth) {
  auto lhsType = cast<RankedTensorType>(lhs.getType());
  auto resultType = RankedTensorType::get(
      {lhsType.getDimSize(0), outputWidth}, lhsType.getElementType());
  SmallVector<Value> dims;
  if (ShapedType::isDynamic(lhsType.getDimSize(0)))
    dims.push_back(rewriter.create<tensor::DimOp>(loc, lhs, 0));
  Value init = createFilled(rewriter, loc, resultType,
                            constantF32(rewriter, loc, 0.0), dims);
  MLIRContext *ctx = rewriter.getContext();
  AffineExpr b, g, k;
  bindDims(ctx, b, g, k);
  SmallVector<AffineMap> maps = {
      AffineMap::get(3, 0, {b, k}, ctx),
      AffineMap::get(3, 0, {g, k}, ctx),
      AffineMap::get(3, 0, {b, g}, ctx)};
  SmallVector<utils::IteratorType> iterators = {
      utils::IteratorType::parallel, utils::IteratorType::parallel,
      utils::IteratorType::reduction};
  return createGeneric(
      rewriter, loc, ValueRange{lhs, rhs}, init, maps, iterators,
      [&](OpBuilder &bld, Location bodyLoc, ValueRange args) {
        Value mul = bld.create<arith::MulFOp>(bodyLoc, args[0], args[1]);
        bld.create<linalg::YieldOp>(
            bodyLoc, bld.create<arith::AddFOp>(bodyLoc, args[2], mul));
      });
}

struct GruCellLowering : OpConversionPattern<eot::GruCellOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(eot::GruCellOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto hType = cast<RankedTensorType>(adaptor.getHPrev().getType());
    int64_t h = hType.getDimSize(1);
    if (ShapedType::isDynamic(h))
      return rewriter.notifyMatchFailure(op, "dynamic hidden size is unsupported");
    Value xProjection = lowerMatmulProjection(
        rewriter, loc, adaptor.getX(), adaptor.getWeightIh(), 3 * h);
    Value hProjection = lowerMatmulProjection(
        rewriter, loc, adaptor.getHPrev(), adaptor.getWeightHh(), 3 * h);
    auto resultType = cast<RankedTensorType>(op.getHNext().getType());
    SmallVector<Value> dims;
    if (ShapedType::isDynamic(resultType.getDimSize(0)))
      dims.push_back(rewriter.create<tensor::DimOp>(loc, adaptor.getX(), 0));
    if (ShapedType::isDynamic(resultType.getDimSize(1)))
      dims.push_back(rewriter.create<arith::ConstantIndexOp>(loc, h));
    Value output = createEmpty(rewriter, loc, resultType, dims);
    MLIRContext *ctx = rewriter.getContext();
    AffineExpr b, d;
    bindDims(ctx, b, d);
    AffineExpr hOffset = getAffineConstantExpr(h, ctx);
    AffineExpr nOffset = getAffineConstantExpr(2 * h, ctx);
    SmallVector<AffineMap> maps = {
        AffineMap::get(2, 0, {b, d}, ctx),
        AffineMap::get(2, 0, {b, d + hOffset}, ctx),
        AffineMap::get(2, 0, {b, d + nOffset}, ctx),
        AffineMap::get(2, 0, {b, d}, ctx),
        AffineMap::get(2, 0, {b, d + hOffset}, ctx),
        AffineMap::get(2, 0, {b, d + nOffset}, ctx),
        AffineMap::get(2, 0, {d}, ctx),
        AffineMap::get(2, 0, {d + hOffset}, ctx),
        AffineMap::get(2, 0, {d + nOffset}, ctx),
        AffineMap::get(2, 0, {d}, ctx),
        AffineMap::get(2, 0, {d + hOffset}, ctx),
        AffineMap::get(2, 0, {d + nOffset}, ctx),
        AffineMap::get(2, 0, {b, d}, ctx),
        AffineMap::get(2, 0, {b, d}, ctx)};
    SmallVector<utils::IteratorType> iterators(2,
                                               utils::IteratorType::parallel);
    Value result = createGeneric(
        rewriter, loc,
        ValueRange{xProjection, xProjection, xProjection, hProjection,
                   hProjection, hProjection, adaptor.getBiasIh(),
                   adaptor.getBiasIh(), adaptor.getBiasIh(),
                   adaptor.getBiasHh(), adaptor.getBiasHh(),
                   adaptor.getBiasHh(), adaptor.getHPrev()},
        output, maps, iterators,
        [&](OpBuilder &bld, Location bodyLoc, ValueRange a) {
          auto sigmoid = [&](Value value) -> Value {
            Value neg = bld.create<arith::NegFOp>(bodyLoc, value);
            Value exp = bld.create<math::ExpOp>(bodyLoc, neg);
            Value denom = bld.create<arith::AddFOp>(
                bodyLoc, constantF32(bld, bodyLoc, 1.0), exp);
            return bld.create<arith::DivFOp>(
                bodyLoc, constantF32(bld, bodyLoc, 1.0), denom);
          };
          Value rInput = bld.create<arith::AddFOp>(
              bodyLoc, bld.create<arith::AddFOp>(bodyLoc, a[0], a[6]),
              bld.create<arith::AddFOp>(bodyLoc, a[3], a[9]));
          Value zInput = bld.create<arith::AddFOp>(
              bodyLoc, bld.create<arith::AddFOp>(bodyLoc, a[1], a[7]),
              bld.create<arith::AddFOp>(bodyLoc, a[4], a[10]));
          Value r = sigmoid(rInput);
          Value z = sigmoid(zInput);
          Value recurrentNew =
              bld.create<arith::AddFOp>(bodyLoc, a[5], a[11]);
          Value newInput = bld.create<arith::AddFOp>(
              bodyLoc, bld.create<arith::AddFOp>(bodyLoc, a[2], a[8]),
              bld.create<arith::MulFOp>(bodyLoc, r, recurrentNew));
          Value candidate = bld.create<math::TanhOp>(bodyLoc, newInput);
          Value oneMinusZ = bld.create<arith::SubFOp>(
              bodyLoc, constantF32(bld, bodyLoc, 1.0), z);
          Value next = bld.create<arith::AddFOp>(
              bodyLoc,
              bld.create<arith::MulFOp>(bodyLoc, oneMinusZ, candidate),
              bld.create<arith::MulFOp>(bodyLoc, z, a[12]));
          bld.create<linalg::YieldOp>(bodyLoc, next);
        });
    rewriter.replaceOp(op, result);
    return success();
  }
};

static Value reduceRows(ConversionPatternRewriter &rewriter, Location loc,
                        Value input, RankedTensorType resultType,
                        double initial, bool maximum) {
  SmallVector<Value> dims;
  if (ShapedType::isDynamic(resultType.getDimSize(0)))
    dims.push_back(rewriter.create<tensor::DimOp>(loc, input, 0));
  Value init = createFilled(rewriter, loc, resultType,
                            constantF32(rewriter, loc, initial), dims);
  MLIRContext *ctx = rewriter.getContext();
  AffineExpr b, c;
  bindDims(ctx, b, c);
  SmallVector<AffineMap> maps = {
      AffineMap::get(2, 0, {b, c}, ctx),
      AffineMap::get(2, 0, {b, getAffineConstantExpr(0, ctx)}, ctx)};
  SmallVector<utils::IteratorType> iterators = {
      utils::IteratorType::parallel, utils::IteratorType::reduction};
  return createGeneric(
      rewriter, loc, ValueRange{input}, init, maps, iterators,
      [&](OpBuilder &bld, Location bodyLoc, ValueRange args) {
        Value value;
        if (maximum)
          value = bld.create<arith::MaximumFOp>(bodyLoc, args[1], args[0]);
        else
          value = bld.create<arith::AddFOp>(bodyLoc, args[1], args[0]);
        bld.create<linalg::YieldOp>(bodyLoc, value);
      });
}

struct GmParameterizeLowering
    : OpConversionPattern<eot::GmParameterizeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(eot::GmParameterizeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto piType = cast<RankedTensorType>(op.getPi().getType());
    auto oneType = cast<RankedTensorType>(op.getTau().getType());
    auto cType = cast<RankedTensorType>(op.getCMean().getType());
    Value batch;
    if (ShapedType::isDynamic(piType.getDimSize(0)) ||
        ShapedType::isDynamic(piType.getDimSize(1)) ||
        ShapedType::isDynamic(oneType.getDimSize(0)) ||
        ShapedType::isDynamic(oneType.getDimSize(1)) ||
        ShapedType::isDynamic(cType.getDimSize(0)) ||
        ShapedType::isDynamic(cType.getDimSize(1)))
      batch = rewriter.create<tensor::DimOp>(loc, adaptor.getPiLogits(), 0);
    auto resultDims = [&](RankedTensorType type, int64_t width) {
      SmallVector<Value> dims;
      if (ShapedType::isDynamic(type.getDimSize(0)))
        dims.push_back(batch);
      if (ShapedType::isDynamic(type.getDimSize(1)))
        dims.push_back(rewriter.create<arith::ConstantIndexOp>(loc, width));
      return dims;
    };
    SmallVector<Value> piDims = resultDims(piType, 4);
    SmallVector<Value> oneDims = resultDims(oneType, 1);
    SmallVector<Value> cDims = resultDims(cType, 2);
    Value rowMax = reduceRows(rewriter, loc, adaptor.getPiLogits(), oneType,
                              -std::numeric_limits<float>::infinity(), true);
    MLIRContext *ctx = rewriter.getContext();
    AffineExpr b, c;
    bindDims(ctx, b, c);
    AffineMap identity = AffineMap::get(2, 0, {b, c}, ctx);
    AffineMap broadcast = AffineMap::get(
        2, 0, {b, getAffineConstantExpr(0, ctx)}, ctx);
    SmallVector<utils::IteratorType> parallel(2,
                                              utils::IteratorType::parallel);
    auto elementwise = [&](ValueRange inputs, RankedTensorType type,
                           ValueRange dims, ArrayRef<AffineMap> inputMaps,
                           auto body) -> Value {
      Value out = createEmpty(rewriter, loc, type, dims);
      SmallVector<AffineMap> maps(inputMaps);
      maps.push_back(rewriter.getMultiDimIdentityMap(2));
      return createGeneric(rewriter, loc, inputs, out, maps, parallel, body);
    };
    Value expLogits = elementwise(
        ValueRange{adaptor.getPiLogits(), rowMax}, piType, piDims,
        ArrayRef<AffineMap>{identity, broadcast},
        [&](OpBuilder &bld, Location bodyLoc, ValueRange a) {
          Value shifted = bld.create<arith::SubFOp>(bodyLoc, a[0], a[1]);
          bld.create<linalg::YieldOp>(
              bodyLoc, bld.create<math::ExpOp>(bodyLoc, shifted));
        });
    Value expSum = reduceRows(rewriter, loc, expLogits, oneType, 0.0, false);
    Value pi = elementwise(
        ValueRange{expLogits, expSum}, piType, piDims,
        ArrayRef<AffineMap>{identity, broadcast},
        [&](OpBuilder &bld, Location bodyLoc, ValueRange a) {
          bld.create<linalg::YieldOp>(
              bodyLoc, bld.create<arith::DivFOp>(bodyLoc, a[0], a[1]));
        });
    Value tau = elementwise(
        ValueRange{adaptor.getTauLogits(), adaptor.getQN()}, oneType, oneDims,
        ArrayRef<AffineMap>{identity, identity},
        [&](OpBuilder &bld, Location bodyLoc, ValueRange a) {
          Value one = constantF32(bld, bodyLoc, 1.0);
          Value sigmoid = bld.create<arith::DivFOp>(
              bodyLoc, one,
              bld.create<arith::AddFOp>(
                  bodyLoc, one,
                  bld.create<math::ExpOp>(
                      bodyLoc, bld.create<arith::NegFOp>(bodyLoc, a[0]))));
          Value scaled = bld.create<arith::MulFOp>(
              bodyLoc, a[1],
              bld.create<arith::MulFOp>(
                  bodyLoc,
                  constantF32(bld, bodyLoc,
                              op.getTauMaxAttr().getValueAsDouble()),
                  sigmoid));
          bld.create<linalg::YieldOp>(
              bodyLoc,
              bld.create<arith::AddFOp>(
                  bodyLoc,
                  constantF32(bld, bodyLoc,
                              op.getTauMinAttr().getValueAsDouble()),
                  scaled));
        });
    Value alphaPi = elementwise(
        ValueRange{pi, tau}, piType, piDims,
        ArrayRef<AffineMap>{identity, broadcast},
        [&](OpBuilder &bld, Location bodyLoc, ValueRange a) {
          Value value = bld.create<arith::AddFOp>(
              bodyLoc,
              constantF32(bld, bodyLoc,
                          op.getAlphaBaseAttr().getValueAsDouble()),
              bld.create<arith::MulFOp>(bodyLoc, a[0], a[1]));
          bld.create<linalg::YieldOp>(bodyLoc, value);
        });
    Value alpha0 = reduceRows(rewriter, loc, alphaPi, oneType, 0.0, false);
    double denomValue = std::log1p(
        4.0 * op.getAlphaBaseAttr().getValueAsDouble() +
        op.getTauMinAttr().getValueAsDouble() +
        op.getTauMaxAttr().getValueAsDouble());
    Value sPi = elementwise(
        ValueRange{alpha0}, oneType, oneDims, ArrayRef<AffineMap>{identity},
        [&](OpBuilder &bld, Location bodyLoc, ValueRange a) {
          Value normalized = bld.create<arith::DivFOp>(
              bodyLoc, bld.create<math::Log1pOp>(bodyLoc, a[0]),
              constantF32(bld, bodyLoc, denomValue));
          Value clamped = bld.create<arith::MaximumFOp>(
              bodyLoc, normalized, constantF32(bld, bodyLoc, 0.0));
          clamped = bld.create<arith::MinimumFOp>(
              bodyLoc, clamped, constantF32(bld, bodyLoc, 1.0));
          bld.create<linalg::YieldOp>(bodyLoc, clamped);
        });
    auto sigmoidTransform = [&](Value input, double low, double high) {
      return elementwise(
          ValueRange{input}, cType, cDims, ArrayRef<AffineMap>{identity},
          [&](OpBuilder &bld, Location bodyLoc, ValueRange a) {
            Value one = constantF32(bld, bodyLoc, 1.0);
            Value sigmoid = bld.create<arith::DivFOp>(
                bodyLoc, one,
                bld.create<arith::AddFOp>(
                    bodyLoc, one,
                    bld.create<math::ExpOp>(
                        bodyLoc, bld.create<arith::NegFOp>(bodyLoc, a[0]))));
            Value value = bld.create<arith::AddFOp>(
                bodyLoc, constantF32(bld, bodyLoc, low),
                bld.create<arith::MulFOp>(
                    bodyLoc, constantF32(bld, bodyLoc, high - low), sigmoid));
            bld.create<linalg::YieldOp>(bodyLoc, value);
          });
    };
    Value cMean = sigmoidTransform(adaptor.getCXi(),
                                   op.getCMinAttr().getValueAsDouble(),
                                   op.getCMaxAttr().getValueAsDouble());
    Value cLogvar = sigmoidTransform(
        adaptor.getRawLogvar(), op.getLogvarMinAttr().getValueAsDouble(),
        op.getLogvarMaxAttr().getValueAsDouble());
    Value cVar = elementwise(
        ValueRange{cLogvar}, cType, cDims, ArrayRef<AffineMap>{identity},
        [&](OpBuilder &bld, Location bodyLoc, ValueRange a) {
          bld.create<linalg::YieldOp>(
              bodyLoc, bld.create<math::ExpOp>(bodyLoc, a[0]));
        });
    Value qC = elementwise(
        ValueRange{cVar}, cType, cDims, ArrayRef<AffineMap>{identity},
        [&](OpBuilder &bld, Location bodyLoc, ValueRange a) {
          Value one = constantF32(bld, bodyLoc, 1.0);
          bld.create<linalg::YieldOp>(
              bodyLoc,
              bld.create<arith::DivFOp>(
                  bodyLoc, one,
                  bld.create<arith::AddFOp>(bodyLoc, one, a[0])));
        });
    auto castToResult = [&](Value value, Value declaredResult) -> Value {
      if (value.getType() == declaredResult.getType())
        return value;
      return rewriter.create<tensor::CastOp>(loc, declaredResult.getType(),
                                             value);
    };
    rewriter.replaceOp(
        op, ValueRange{castToResult(pi, op.getPi()),
                       castToResult(tau, op.getTau()),
                       castToResult(alphaPi, op.getAlphaPi()),
                       castToResult(alpha0, op.getAlpha0()),
                       castToResult(sPi, op.getSPi()),
                       castToResult(cMean, op.getCMean()),
                       castToResult(cLogvar, op.getCLogvar()),
                       castToResult(cVar, op.getCVar()),
                       castToResult(qC, op.getQC())});
    return success();
  }
};

struct ConvertEotToStandardPass
    : ::eot::impl::ConvertEotToStandardBase<ConvertEotToStandardPass> {
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    linalg::LinalgDialect, math::MathDialect,
                    scf::SCFDialect, tensor::TensorDialect,
                    tosa::TosaDialect>();
  }

  void runOnOperation() final {
    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect, BuiltinDialect,
                           func::FuncDialect, linalg::LinalgDialect,
                           math::MathDialect, scf::SCFDialect,
                           tensor::TensorDialect, tosa::TosaDialect>();
    target.addLegalOp<ModuleOp>();
    target.addIllegalOp<eot::PointFeaturesOp, eot::MaskedMeanOp,
                        eot::MaskedVarianceOp, eot::MaskedMaxOp,
                        eot::MaskedMinOp, eot::GruCellOp,
                        eot::GmParameterizeOp>();
    RewritePatternSet patterns(&getContext());
    patterns.add<PointFeaturesLowering,
                 MaskedAverageLowering<eot::MaskedMeanOp, false>,
                 MaskedAverageLowering<eot::MaskedVarianceOp, true>,
                 MaskedExtremaLowering<eot::MaskedMaxOp, true>,
                 MaskedExtremaLowering<eot::MaskedMinOp, false>, GruCellLowering,
                 GmParameterizeLowering>(&getContext());
    if (failed(applyFullConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> eot::createConvertEotToStandardPass() {
  return std::make_unique<ConvertEotToStandardPass>();
}
