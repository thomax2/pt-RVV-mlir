#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "npu/NpuDialect.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
// #include "toy/ToyOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <cstdint>
#include <optional>
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "npu/NpuTypes.h"
#define GEN_PASS_DEF_CONVERTTOSATONPU
#include "npu/NpuPasses.h"

#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace llvm;
using namespace npu;

struct Conv2dOpPat : OpConversionPattern<tosa::Conv2DOp> {
    using OpConversionPattern<tosa::Conv2DOp>::OpConversionPattern;
    LogicalResult matchAndRewrite(tosa::Conv2DOp op, tosa::Conv2DOpAdaptor adaptor,ConversionPatternRewriter & rewriter) const override {

        Value input = adaptor.getInput();
        Value weight = adaptor.getWeight();
        Value bias = adaptor.getBias();

        auto pad = rewriter.getI64ArrayAttr(op.getPad());
        auto stride = rewriter.getI64ArrayAttr(op.getStride());
        auto dilation = rewriter.getI64ArrayAttr(op.getDilation());

        // 关键步骤 1：获取 TOSA 的结果类型
        Type originalResultType = op.getType();
        
        // 关键步骤 2：使用 TypeConverter 将其转换为 NPU 类型
        // getTypeConverter() 是 OpConversionPattern 父类提供的方法
        Type convertedType = getTypeConverter()->convertType(originalResultType);

        if (!convertedType) return failure();

        auto npuConv = rewriter.create<npu::Conv2dOp>(
            op->getLoc(),
            convertedType,
            input, weight, bias,
            pad, stride, dilation
        );
        rewriter.replaceOp(op, npuConv.getResult());
        return success();
    }
};

struct ConstantOpPat : OpConversionPattern<tosa::ConstOp> {
    using OpConversionPattern<tosa::ConstOp>::OpConversionPattern;
    LogicalResult matchAndRewrite(tosa::ConstOp op, OpAdaptor adaptor, ConversionPatternRewriter & rewriter) const override {
        // 获取并转换结果类型
        Type convertedType = getTypeConverter()->convertType(op.getType());
        if (!convertedType) return failure();

        rewriter.replaceOpWithNewOp<npu::ConstantOp>(
            op,
            convertedType,
            op.getValuesAttr()
        );
        return success();
    }
};

struct ConstantShapeOpPat : OpConversionPattern<tosa::ConstShapeOp> {
    using OpConversionPattern<tosa::ConstShapeOp>::OpConversionPattern;
    LogicalResult matchAndRewrite(tosa::ConstShapeOp op, OpAdaptor adaptor, ConversionPatternRewriter & rewriter) const override {
        // Value values = adaptor.getValues();

        auto tosaShapeType = op.getType();
        int rank = tosaShapeType.getRank();
        auto npuShapeType = NpuShapeType::get(rewriter.getContext(), rank);

        rewriter.replaceOpWithNewOp<ConstantShapeOp>(op, npuShapeType, op.getValues());
        return success();
    }
};



struct ReshapeOpPat : OpConversionPattern<tosa::ReshapeOp> {
    using OpConversionPattern<tosa::ReshapeOp>::OpConversionPattern;
    LogicalResult matchAndRewrite(tosa::ReshapeOp op, OpAdaptor adaptor, ConversionPatternRewriter & rewriter) const override {
        Value input1 = adaptor.getInput1();
        Value shape = adaptor.getShape();

        Type originalResultType = op.getType();
        Type convertedType = getTypeConverter()->convertType(originalResultType);
        if (!convertedType) return failure();

        auto reshape = rewriter.create<npu::ReshapeOp>(
            op->getLoc(),
            convertedType,
            input1, shape
        );
        rewriter.replaceOp(op, reshape.getResult());
        return success();
    }
};

struct ClampOpPat : OpConversionPattern<tosa::ClampOp> {
    using OpConversionPattern<tosa::ClampOp>::OpConversionPattern;
    LogicalResult matchAndRewrite(tosa::ClampOp op, OpAdaptor adaptor, ConversionPatternRewriter & rewriter) const override {
        Value input = adaptor.getInput();
        Attribute minAttr = op.getMinValAttr();
        Attribute maxAttr = op.getMaxValAttr();

        auto extractValue = [](Attribute attr) -> std::optional<float> {
            if(auto floatAttr = dyn_cast<FloatAttr>(attr)) {
                return static_cast<float>(floatAttr.getValueAsDouble());
            }
            if(auto intAttr = dyn_cast<IntegerAttr>(attr)) {
                return static_cast<float>(intAttr.getInt());
            }
            return std::nullopt;
        };

        std::optional<float> min = extractValue(minAttr);
        std::optional<float> max = extractValue(maxAttr);
        if (!min || !max)
            return rewriter.notifyMatchFailure(op, "unsupported clamp bound attribute type");

        Type convertedType = getTypeConverter()->convertType(op.getType());
        if (!convertedType) return failure();

        rewriter.replaceOpWithNewOp<npu::ClampOp>(
            op,
            convertedType,
            input,
            rewriter.getF32FloatAttr(*min), rewriter.getF32FloatAttr(*max)
        );
        return success();
    }
};

struct MaxPool2dOpPat : OpConversionPattern<tosa::MaxPool2dOp> {
    using OpConversionPattern<tosa::MaxPool2dOp>::OpConversionPattern;
    LogicalResult matchAndRewrite(tosa::MaxPool2dOp op, OpAdaptor adaptor, ConversionPatternRewriter & rewriter) const override {
        Value input = adaptor.getInput();
        // Attribute kernel = op.getKernelAttr();
        // Attribute stride = op.getStrideAttr();
        // Attribute pad = op.getPadAttr();

        auto pad = rewriter.getI64ArrayAttr(op.getPad());
        auto stride = rewriter.getI64ArrayAttr(op.getStride());
        auto kernel = rewriter.getI64ArrayAttr(op.getKernel());

        Type convertedType = getTypeConverter()->convertType(op.getType());
        if (!convertedType) return failure();

        rewriter.replaceOpWithNewOp<npu::MaxPool2dOp>(
            op,
            convertedType,
            input,
            kernel, stride, pad
        );
        return success();
    }
};

struct TransposeOpPat : OpConversionPattern<tosa::TransposeOp> {
    using OpConversionPattern<tosa::TransposeOp>::OpConversionPattern;
    LogicalResult matchAndRewrite(tosa::TransposeOp op, OpAdaptor adaptor, ConversionPatternRewriter & rewriter) const override {
        Value input1 = adaptor.getInput1();
        auto perms = rewriter.getI32ArrayAttr(op.getPerms());

        Type convertedType = getTypeConverter()->convertType(op.getType());
        if (!convertedType) return failure();

        rewriter.replaceOpWithNewOp<npu::TransposeOp>(
            op,
            convertedType,
            input1,
            perms
        );
        return success();
    }
};

struct MatmulOpPat : OpConversionPattern<tosa::MatMulOp> {
    using OpConversionPattern<tosa::MatMulOp>::OpConversionPattern;
    LogicalResult matchAndRewrite(tosa::MatMulOp op, OpAdaptor adaptor, ConversionPatternRewriter & rewriter) const override {
        Value a = adaptor.getA();
        Value b = adaptor.getB();

        Type originalResultType = op.getType();
        Type convertedType = getTypeConverter()->convertType(originalResultType);
        if (!convertedType) return failure();
        
        rewriter.replaceOpWithNewOp<npu::MatmulOp>(op, convertedType, a, b);
        return success();
    }
};

struct AddOpPat : OpConversionPattern<tosa::AddOp> {
    using OpConversionPattern<tosa::AddOp>::OpConversionPattern;
    LogicalResult matchAndRewrite(tosa::AddOp op, OpAdaptor adaptor, ConversionPatternRewriter & rewriter) const override {
        Value a = adaptor.getInput1();
        Value b = adaptor.getInput2();

        Type originalResultType = op.getType();
        Type convertedType = getTypeConverter()->convertType(originalResultType);
        if (!convertedType) return failure();

        rewriter.replaceOpWithNewOp<npu::AddOp>(op, convertedType, a, b);
        return success();
    }
};

struct ReturnOpPat : public OpConversionPattern<func::ReturnOp> {
    using OpConversionPattern<func::ReturnOp>::OpConversionPattern;
  
    LogicalResult matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
      // 这里的 adaptor.getOperands() 会自动获取经过 TypeConverter 转换后的新操作数
      // 即：把原本的 tensor 类型操作数替换为 npu.tensor 类型
      rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
      return success();
    }
  };
  

struct ConvertTosaToNpuPass : ::npu::impl::ConvertTosaToNpuBase<ConvertTosaToNpuPass> {
    using ::npu::impl::ConvertTosaToNpuBase<ConvertTosaToNpuPass>::ConvertTosaToNpuBase;
    void getDependentDialects(DialectRegistry &registry) const final {
      registry.insert<NpuDialect>();
      registry.insert<func::FuncDialect>();
      registry.insert<tosa::TosaDialect>();
    }

    void runOnOperation() final {
        ConversionTarget target(getContext());
        target.addLegalDialect<npu::NpuDialect>();
        target.addIllegalDialect<tosa::TosaDialect>();
        // 允许转换框架生成的（Cast Op）存在
        target.addLegalOp<UnrealizedConversionCastOp>();

        TypeConverter converter;
        converter.addConversion([&](RankedTensorType tosaTensor) -> NpuTensorType {
            ArrayRef<int64_t> shape = tosaTensor.getShape();
            Type elementType = tosaTensor.getElementType();

            return NpuTensorType::get(&getContext(), shape, elementType);
        });

        converter.addConversion([&](tosa::shapeType tosaShape) -> NpuShapeType {
            // llvm::SmallVector<int64_t> shapeDims = {tosaShape.getRank()};
            // Type i64Type = IntegerType::get(&getContext(), 64);
            int rank = tosaShape.getRank();
            return NpuShapeType::get(&getContext(), rank);
        });

        converter.addConversion([&](NpuTensorType type) -> Type {
            return type;
        });

        converter.addConversion([&](NpuShapeType type) -> Type {
            return type;
        });
        // 处理 Source -> Target 的转换 (例如在 Op 内部创建新类型)
        converter.addTargetMaterialization([](OpBuilder& builder, Type resultType, ValueRange inputs, Location loc) -> Value {
            return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs).getResult(0);
        });

        // 【新增】处理 Target -> Source 的转换 
        // 这是修复 func.func 报错的关键：允许将新的 NpuTensor 转回旧的 tensor 给函数体使用
        converter.addSourceMaterialization([](OpBuilder& builder, Type resultType, ValueRange inputs, Location loc) -> Value {
            return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs).getResult(0);
        });


        target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
            return converter.isSignatureLegal(op.getFunctionType());
        });

        target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
            return converter.isLegal(op.getOperandTypes());
        });

        RewritePatternSet patterns(&getContext());

        patterns.add<Conv2dOpPat, ConstantOpPat, ReturnOpPat, ConstantShapeOpPat, ReshapeOpPat, ClampOpPat, MaxPool2dOpPat,
            TransposeOpPat, MatmulOpPat, AddOpPat>(converter, &getContext());

        populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, converter);
        
        if(failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
            signalPassFailure();
    }
};

std::unique_ptr<mlir::Pass> npu::createConvertTosaToNpuPass(ConvertTosaToNpuOptions options) {
    return std::make_unique<ConvertTosaToNpuPass>(options);
}

