#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "npu/NpuDialect.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
// #include "toy/ToyOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <cstdint>
#include <optional>
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "npu/NpuOps.h"
#include "npu/NpuTypes.h"
#define GEN_PASS_DEF_CONVERTTOSATONPU
#define GEN_PASS_DEF_CONVERTNPUTOVECSCF
#include "npu/NpuPasses.h"

#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace llvm;
using namespace npu;

struct Conv2dOpPatNew : OpConversionPattern<npu::Conv2dOp> {
    using OpConversionPattern<npu::Conv2dOp>::OpConversionPattern;
    LogicalResult matchAndRewrite(npu::Conv2dOp op, OpAdaptor adaptor,ConversionPatternRewriter & rewriter) const override {
        // 1. 获取操作数
        Value input = adaptor.getInput();
        Value weight = adaptor.getWeight();
        Value bias = adaptor.getBias();
        Location loc = op.getLoc();

        // 2. 获取类型信息
        auto inputType = cast<MemRefType>(input.getType());
        auto weightType = cast<MemRefType>(weight.getType());
        auto resultType = cast<MemRefType>(getTypeConverter()->convertType(op.getType()));
        auto elementType = resultType.getElementType();

        // 3. Alloc 输出内存
        Value output = rewriter.create<memref::AllocOp>(loc, resultType);

        // 4. 解析属性 (Pad, Stride, Dilation)
        auto getIntArray = [](ArrayAttr attr) -> SmallVector<int64_t> {
            SmallVector<int64_t> res;
            if (!attr) return res;
            for (auto val : attr.getValue()) {
                res.push_back(cast<IntegerAttr>(val).getInt());
            }
            return res;
        };

        SmallVector<int64_t> pads = getIntArray(op.getPad());       // [top, bottom, left, right]
        SmallVector<int64_t> strides = getIntArray(op.getStride()); // [sh, sw]
        SmallVector<int64_t> dilations = getIntArray(op.getDilation()); // [dh, dw]

        int64_t padTop = pads[0];
        int64_t padLeft = pads[2];
        int64_t strideH = strides[0];
        int64_t strideW = strides[1];
        int64_t dilationH = dilations[0];
        int64_t dilationW = dilations[1];

        // 获取 Kernel 大小 (从 Weight Shape 获取: [OC, KH, KW, IC])
        int64_t KH = weightType.getDimSize(1);
        int64_t KW = weightType.getDimSize(2);
        int64_t IC = weightType.getDimSize(3);

        SmallVector<Value> lowerBounds, upperBounds, steps;
        for(int64_t dimSize : resultType.getShape()) {
            lowerBounds.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
            upperBounds.push_back(rewriter.create<arith::ConstantIndexOp>(loc, dimSize));
            steps.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 1));
        }

        // --- 准备常量 Index 用于计算 ---
        Value c_strideH = rewriter.create<arith::ConstantIndexOp>(loc, strideH);
        Value c_strideW = rewriter.create<arith::ConstantIndexOp>(loc, strideW);
        Value c_dilationH = rewriter.create<arith::ConstantIndexOp>(loc, dilationH);
        Value c_dilationW = rewriter.create<arith::ConstantIndexOp>(loc, dilationW);
        Value c_padTop = rewriter.create<arith::ConstantIndexOp>(loc, padTop);
        Value c_padLeft = rewriter.create<arith::ConstantIndexOp>(loc, padLeft);
        Value H_in_dim = rewriter.create<arith::ConstantIndexOp>(loc, inputType.getDimSize(1));
        Value W_in_dim = rewriter.create<arith::ConstantIndexOp>(loc, inputType.getDimSize(2));

        rewriter.create<scf::ParallelOp>(
            loc, lowerBounds, upperBounds, steps,
            [&](OpBuilder &builder, Location loc, ValueRange ivs) {
                // ivs = [b, oh, ow, oc], output parallel
                Value b_idx = ivs[0];
                Value oh_idx = ivs[1];
                Value ow_idx = ivs[2];
                Value oc_idx = ivs[3];

                // --- 初始化 Accumulator ---
                // 技巧：直接把 Bias 的值读出来作为初始值，省去最后再做一次 Add 操作
                Value initAcc = builder.create<memref::LoadOp>(loc, bias, ValueRange{oc_idx});


                // --- 计算锚点 (Anchor Point) ---
                // h_base = oh * stride_h - pad_top
                Value h_base = builder.create<arith::MulIOp>(loc, oh_idx, c_strideH);
                h_base = builder.create<arith::SubIOp>(loc, h_base, c_padTop);

                // w_base = ow * stride_w - pad_left
                Value w_base = builder.create<arith::MulIOp>(loc, ow_idx, c_strideW);
                w_base = builder.create<arith::SubIOp>(loc, w_base, c_padLeft);
                
                // 6. 内层归约循环: [KH, KW, IC]
                // 必须使用 scf::ForOp + iter_args 来做累加
                auto loopKH = builder.create<scf::ForOp>(
                    loc,
                    builder.create<arith::ConstantIndexOp>(loc, 0),
                    builder.create<arith::ConstantIndexOp>(loc, KH),
                    builder.create<arith::ConstantIndexOp>(loc, 1),
                    ValueRange{initAcc},
                    [&](OpBuilder &b2, Location l2, Value kh_idx, ValueRange argsH) {
                        Value accH = argsH[0];

                        auto loopKW = b2.create<scf::ForOp>(
                            l2,
                            b2.create<arith::ConstantIndexOp>(l2, 0),
                            b2.create<arith::ConstantIndexOp>(l2, KW),
                            b2.create<arith::ConstantIndexOp>(l2, 1),
                            ValueRange{accH},
                            [&](OpBuilder &b3, Location l3, Value kw_idx, ValueRange argsW) {
                                Value accW = argsW[0];
                                
                                auto loopIC = b3.create<scf::ForOp>(
                                    l3,
                                    b3.create<arith::ConstantIndexOp>(l3, 0),
                                    b3.create<arith::ConstantIndexOp>(l3, IC),
                                    b3.create<arith::ConstantIndexOp>(l3, 1),
                                    ValueRange{accW},
                                    [&](OpBuilder &b4, Location l4, Value ic_idx, ValueRange argsIC) {
                                        Value currentAcc = argsIC[0];

                                        // --- 坐标计算 (包含 Dilation) ---
                                        // h_in = h_base + kh * dilation_h
                                        Value kh_dilated = b4.create<arith::MulIOp>(l4, kh_idx, c_dilationH);
                                        Value h_in = b4.create<arith::AddIOp>(l4, h_base, kh_dilated);

                                        // w_in = w_base + kw * dilation_w
                                        Value kw_dilated = b4.create<arith::MulIOp>(l4, kw_idx, c_dilationW);
                                        Value w_in = b4.create<arith::AddIOp>(l4, w_base, kw_dilated);

                                        // --- 边界检查 ---
                                        // 只需要检查 h_in 和 w_in 是否在 Input 的范围内

                                        Value h_ge_0 = b4.create<arith::CmpIOp>(l4, arith::CmpIPredicate::sge, h_in, b4.create<arith::ConstantIndexOp>(l4, 0));
                                        Value h_lt_H = b4.create<arith::CmpIOp>(l4, arith::CmpIPredicate::slt, h_in, H_in_dim);
                                        Value w_ge_0 = b4.create<arith::CmpIOp>(l4, arith::CmpIPredicate::sge, w_in, b4.create<arith::ConstantIndexOp>(l4, 0));
                                        Value w_lt_W = b4.create<arith::CmpIOp>(l4, arith::CmpIPredicate::slt, w_in, W_in_dim);

                                        Value validH = b4.create<arith::AndIOp>(l4, h_ge_0, h_lt_H);
                                        Value validW = b4.create<arith::AndIOp>(l4, w_ge_0, w_lt_W);
                                        Value isValid = b4.create<arith::AndIOp>(l4, validH, validW);

                                        auto ifOp = b4.create<scf::IfOp>(
                                            l4, elementType, isValid, /*withElseRegion=*/true
                                        );

                                        // Then: 有效区域 -> Load Input * Weight
                                        {
                                            OpBuilder thenB = ifOp.getThenBodyBuilder();
                                            // Load Input [b, h_in, w_in, ic]
                                            Value valIn = thenB.create<memref::LoadOp>(l4, input, ValueRange{b_idx, h_in, w_in, ic_idx});
                                            // Load Weight [oc, kh, kw, ic]
                                            Value valWt = thenB.create<memref::LoadOp>(l4, weight, ValueRange{oc_idx, kh_idx, kw_idx, ic_idx});
                                            
                                            Value prod;
                                            if (elementType.isF32()) prod = thenB.create<arith::MulFOp>(l4, valIn, valWt);
                                            else prod = thenB.create<arith::MulIOp>(l4, valIn, valWt);
                                            
                                            thenB.create<scf::YieldOp>(l4, prod);
                                        }

                                        // Else: Padding 区域 -> 0
                                        {
                                            OpBuilder elseB = ifOp.getElseBodyBuilder();
                                            Value zero;
                                            if (elementType.isF32()) zero = elseB.create<arith::ConstantOp>(l4, FloatAttr::get(elementType, 0.0));
                                            else zero = elseB.create<arith::ConstantOp>(l4, IntegerAttr::get(elementType, 0));
                                            elseB.create<scf::YieldOp>(l4, zero);
                                        }

                                        // --- 累加 ---
                                        Value prodVal = ifOp.getResult(0);
                                        Value newAcc;
                                        if (elementType.isF32()) newAcc = b4.create<arith::AddFOp>(l4, currentAcc, prodVal);
                                        else newAcc = b4.create<arith::AddIOp>(l4, currentAcc, prodVal);

                                        b4.create<scf::YieldOp>(l4, newAcc);
                                    }
                                );
                                b3.create<scf::YieldOp>(l3, loopIC.getResult(0));
                            }
                        );
                        b2.create<scf::YieldOp>(l2, loopKW.getResult(0));
                    }
                );
                // 7. Store Result
                Value finalRes = loopKH.getResult(0);
                builder.create<memref::StoreOp>(loc, finalRes, output, ivs); // ivs = [b, oh, ow, oc]
                // builder.create<scf::YieldOp>(loc);
                builder.create<scf::ReduceOp>(loc);
            }
        );
        rewriter.replaceOp(op, output);
        return success();
    }
};


struct ConstantOpPatNew : OpConversionPattern<npu::ConstantOp> {
    using OpConversionPattern<npu::ConstantOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(npu::ConstantOp op, OpAdaptor adaptor, ConversionPatternRewriter & rewriter) const override {        
        // 1. 获取原始属性
        auto elementsAttr = adaptor.getValuesAttr();
        
        // 2. 使用类型转换器获取转换后的类型
        Type originalResultType = op.getType();
        Type convertedType = getTypeConverter()->convertType(originalResultType);
        if (!convertedType) {
            llvm::errs() << "TypeConverter failed to convert type!\n";
            return failure();
        }
        auto memrefType = cast<MemRefType>(convertedType);

        // 3. 生成全局变量名
        std::string globalName = "npu_const_" + std::to_string((uintptr_t)op.getOperation());

        // 4. 找到 ModuleOp (全局变量必须插在 Module 里，不能插在 Func 里)
        auto module = op->getParentOfType<ModuleOp>();

        // 使用 InsertionGuard 保存插入点，因为我们要跳到 Module Body 也就是函数外面去写代码
        {
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(module.getBody());

            // 5. 创建 memref.global
            // memref.global 会自动接受 tensor 类型的 ElementsAttr 作为初始值
            rewriter.create<memref::GlobalOp>(
                op.getLoc(),
                globalName,
                rewriter.getStringAttr("private"), // linkage
                memrefType,
                elementsAttr, // <--- 关键！直接把 dense/dense_resource 传进去
                /*constant=*/true,
                /*alignment=*/nullptr
            );
        }

        // 6. 在原位置创建 memref.get_global
        // 这将替换原来的 npu.const，返回一个 memref 值
        rewriter.replaceOpWithNewOp<memref::GetGlobalOp>(
            op,
            memrefType,
            globalName
        );

        return success();
    }
};



struct ConstantShapeOpPatNew : OpConversionPattern<npu::ConstantShapeOp> {
    using OpConversionPattern<npu::ConstantShapeOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(npu::ConstantShapeOp op, OpAdaptor adaptor, ConversionPatternRewriter & rewriter) const override {
        // 1. 获取原始属性
        auto elementsAttr = adaptor.getValuesAttr();

        // 2. 使用类型转换器获取转换后的类型
        Type originalResultType = op.getType();
        Type convertedType = getTypeConverter()->convertType(originalResultType);
        if (!convertedType) {
            llvm::errs() << "TypeConverter failed to convert type!\n";
            return failure();
        }
        auto memrefType = cast<MemRefType>(convertedType);

        // 3. 生成全局变量名
        std::string globalName = "npu_const_shape_" + std::to_string((uintptr_t)op.getOperation());

        // 4. 找到 ModuleOp (全局变量必须插在 Module 里，不能插在 Func 里)
        auto module = op->getParentOfType<ModuleOp>();

        // 使用 InsertionGuard 保存插入点，因为我们要跳到 Module Body 也就是函数外面去写代码
        {
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(module.getBody());

            // 5. 创建 memref.global
            // memref.global 会自动接受 tensor 类型的 ElementsAttr 作为初始值
            rewriter.create<memref::GlobalOp>(
                op.getLoc(),
                globalName,
                rewriter.getStringAttr("private"), // linkage
                memrefType,
                elementsAttr, // <--- 关键！直接把 dense/dense_resource 传进去
                /*constant=*/true,
                /*alignment=*/nullptr
            );
        }

        // 6. 在原位置创建 memref.get_global
        // 这将替换原来的 npu.const，返回一个 memref 值
        rewriter.replaceOpWithNewOp<memref::GetGlobalOp>(
            op,
            memrefType,
            globalName
        );

        return success();
    }
};


struct ReshapeOpPatNew : OpConversionPattern<npu::ReshapeOp> {
    using OpConversionPattern<npu::ReshapeOp>::OpConversionPattern;
    LogicalResult matchAndRewrite(npu::ReshapeOp op, OpAdaptor adaptor, ConversionPatternRewriter & rewriter) const override {
        Value input = adaptor.getInput();
        // auto loc = op->getLoc();

        Type originalResultType = op.getType();
        Type convertedType = getTypeConverter()->convertType(originalResultType);
        auto resultType = cast<MemRefType>(convertedType);
        
        // 目前仅支持静态 Shape (Compile-time known shape)
        // 动态 Shape 需要生成 arith 指令在运行时计算，逻辑会极其复杂
        if(!resultType.hasStaticShape()) {
            return rewriter.notifyMatchFailure(op, "Dynamic shape not supported in this pattern yet.");
        }

        // 2. 准备 ReinterpretCast 需要的三个参数：Offset, Sizes, Strides
        OpFoldResult offset = rewriter.getIndexAttr(0); 

        SmallVector<OpFoldResult> sizes, strides;
        auto shape = resultType.getShape();
        for(int64_t dimSize : shape)
            sizes.push_back(rewriter.getIndexAttr(dimSize));

        // --- C. Strides (关键步骤) ---
        // 我们需要计算标准的“行优先”步长 (Row-Major Strides)
        // 规则：最后一个维度的 stride 是 1，前面的 stride = 后一维 stride * 后一维 size
        // 例如 Shape [2, 3, 4] -> Strides [12, 4, 1]

        SmallVector<int64_t, 4> tmpStride(shape.size());
        int currentStride = 1;

        for(int i = shape.size() - 1; i >=0; i--) {
            tmpStride[i] = currentStride;
            currentStride *= shape[i];
        }

        for(int64_t stride : tmpStride)
            strides.push_back(rewriter.getIndexAttr(stride));

        rewriter.replaceOpWithNewOp<memref::ReinterpretCastOp>(
            op, resultType,
            input,
            offset, sizes, strides);

        return success();
    }
};

struct ClampOpPatNew : OpConversionPattern<npu::ClampOp> {
    using OpConversionPattern<npu::ClampOp>::OpConversionPattern;
    LogicalResult matchAndRewrite(npu::ClampOp op, OpAdaptor adaptor, ConversionPatternRewriter & rewriter) const override {
        Value input = adaptor.getInput();
        auto loc = op->getLoc();

        // alloc memory
        Type originalResultType = op.getType();
        Type convertedType = getTypeConverter()->convertType(originalResultType);
        auto resultType = cast<MemRefType>(convertedType);

        auto output = rewriter.create<memref::AllocOp>(loc, resultType);


        // get value
        auto elementType = resultType.getElementType();  // <--- 关键！这是 f32 或 i32
        Value minVal, maxVal;

        Attribute minAttr = op.getMinValAttr();
        Attribute maxAttr = op.getMaxValAttr();

        auto extractValue = [](Attribute attr) -> float {
            if(auto floatAttr = dyn_cast<FloatAttr>(attr)) {
                return (float)floatAttr.getValueAsDouble();
            }
            if(auto intAttr = dyn_cast<IntegerAttr>(attr)) {
                return (float)intAttr.getInt();
            }
            llvm_unreachable("Unsupported attribute type for min/max value");
        };

        float min = extractValue(minAttr);
        float max = extractValue(maxAttr);

        // 创建标量常量
        // 注意：FloatAttr::get 的第一个参数必须是 elementType (如 f32)，不能是 tensor type
        minVal = rewriter.create<arith::ConstantOp>(loc, FloatAttr::get(elementType, min));
        maxVal = rewriter.create<arith::ConstantOp>(loc, FloatAttr::get(elementType, max));


        auto shape = resultType.getShape();
        SmallVector<Value> lowerBounds(shape.size(), rewriter.create<arith::ConstantIndexOp>(loc, 0));
        SmallVector<Value> upperBounds;
        SmallVector<Value> steps(shape.size(), rewriter.create<arith::ConstantIndexOp>(loc, 1));
        
        for(int64_t dimSize : shape) {
            upperBounds.push_back(rewriter.create<arith::ConstantIndexOp>(loc, dimSize));
        }

        rewriter.create<scf::ParallelOp>(
            loc,
            lowerBounds,
            upperBounds,
            steps,
            [&](OpBuilder &Builder, Location Loc, ValueRange localIvs) {

                Value lhs = Builder.create<memref::LoadOp>(Loc, input, localIvs);
                Value clamped;
                if (isa<FloatType>(elementType)) {
                    // 浮点数 clamp: min(max, max(min, x))
                    Value maxMin = Builder.create<arith::MaximumFOp>(Loc, lhs, minVal);
                    clamped = Builder.create<arith::MinimumFOp>(Loc, maxMin, maxVal);
                } else if (elementType.isSignedInteger()) {
                    // 有符号整数 clamp
                    Value maxMin = Builder.create<arith::MaxSIOp>(Loc, lhs, minVal);
                    clamped = Builder.create<arith::MinSIOp>(Loc, maxMin, maxVal);
                } else if (elementType.isUnsignedInteger()) {
                    // 无符号整数 clamp
                    Value maxMin = Builder.create<arith::MaxUIOp>(Loc, lhs, minVal);
                    clamped = Builder.create<arith::MinUIOp>(Loc, maxMin, maxVal);
                } else {
                    // 不应该到达这里，因为上面已经检查了类型
                    llvm_unreachable("clamped type wrong");    
                    return;
                }
                Builder.create<memref::StoreOp>(Loc, clamped, output, localIvs);
                Builder.create<scf::ReduceOp>(loc);
            }
        );
        rewriter.replaceOp(op, output);
        return success();
    }
};


struct MaxPool2dOpPatNew : OpConversionPattern<npu::MaxPool2dOp> {
    using OpConversionPattern<npu::MaxPool2dOp>::OpConversionPattern;
    LogicalResult matchAndRewrite(npu::MaxPool2dOp op, OpAdaptor adaptor, ConversionPatternRewriter & rewriter) const override {
        Value input = adaptor.getInput();
        auto loc = op->getLoc();

        // 1. 获取输入输出类型
        auto inputType = cast<MemRefType>(input.getType());
        auto resultType = cast<MemRefType>(getTypeConverter()->convertType(op.getType()));
        auto elementType = resultType.getElementType();

        Value output = rewriter.create<memref::AllocOp>(loc, resultType);

        // 3. 解析属性
        // ArrayAttr 里面存的是 Attribute 基类，需要转换成 IntegerAttr 再取值
        // Helper lambda to extract int64_t from ArrayAttr
        auto getIntArray = [](ArrayAttr attr) -> SmallVector<int64_t> {
            SmallVector<int64_t> res;
            if (!attr) return res; 
            for (auto val : attr.getValue()) {
                res.push_back(cast<IntegerAttr>(val).getInt());
            }
            return res;
        };

        SmallVector<int64_t> pads = getIntArray(op.getPad());       // [top, bottom, left, right]  Size: 4
        SmallVector<int64_t> strides = getIntArray(op.getStride()); // Size: 2
        SmallVector<int64_t> kernels = getIntArray(op.getKernel()); // Size: 2
        
        // 假设顺序是: pads=[pad_h, pad_w, ...], strides=[s_h, s_w], kernels=[k_h, k_w]
        // 且 tensor 格式为 NHWC (N, H, W, C)
        int64_t padTop  = pads[0]; 
        int64_t padLeft = pads[2];
        int64_t strideH = strides[0];
        int64_t strideW = strides[1];
        int64_t kernelH = kernels[0];
        int64_t kernelW = kernels[1];

        Value initVal;
        if (elementType.isF32()) {
            initVal = rewriter.create<arith::ConstantOp>(loc, FloatAttr::get(elementType, -std::numeric_limits<float>::infinity()));
        } else if (elementType.isInteger()) {
            initVal = rewriter.create<arith::ConstantOp>(loc, IntegerAttr::get(elementType, std::numeric_limits<int64_t>::min()));
        } else {
             return failure(); // Handle other types
        }

        SmallVector<Value> lowerBounds, upperBounds, steps;

        for (int64_t dimSize : resultType.getShape()) {
            // 2. 使用 create<arith::ConstantIndexOp> (而不是 getIndexAttr)
            lowerBounds.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
            upperBounds.push_back(rewriter.create<arith::ConstantIndexOp>(loc, dimSize));
            steps.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 1));
        }


        rewriter.create<scf::ParallelOp>(
            loc,
            lowerBounds,
            upperBounds,
            steps,
            [&](OpBuilder &Builder, Location Loc, ValueRange localIvs) {
                Value n = localIvs[0];
                Value h_out = localIvs[1];
                Value w_out = localIvs[2];
                Value c = localIvs[3];

                // 当前 Max 值初始化
                // 因为 scf.parallel 内部不好做复杂 reduction，我们在这里开辟一个小内存或者用 scf.for 迭代更新一个 var
                // 为了简单，我们使用 stack 上的 MemRef (alloca) 或者直接 reduce
                // 最简单方法：生成一个 MemRef 来存临时 max，或者只用 scf.for + iter_args (推荐)
                // 这里我们使用 scf.for 来遍历 Kernel 窗口，并携带 max 值

                Value finalMax = initVal;

                // h_base = h_out * stride_h - pad_top
                Value h_base = Builder.create<arith::MulIOp>(Loc, h_out, Builder.create<arith::ConstantIndexOp>(Loc, strideH));
                h_base = Builder.create<arith::SubIOp>(Loc, h_base, Builder.create<arith::ConstantIndexOp>(Loc, padTop)); // 使用 padTop

                // w_base = w_out * stride_w - pad_left
                Value w_base = Builder.create<arith::MulIOp>(Loc, w_out, Builder.create<arith::ConstantIndexOp>(Loc, strideW));
                w_base = Builder.create<arith::SubIOp>(Loc, w_base, Builder.create<arith::ConstantIndexOp>(Loc, padLeft)); // 使用 padLeft
                
                // 用 scf.for 遍历 kernelH / kernelW
                auto loopH = Builder.create<scf::ForOp>(
                    Loc,
                    Builder.create<arith::ConstantIndexOp>(loc, 0), 
                    Builder.create<arith::ConstantIndexOp>(loc, kernelH), 
                    Builder.create<arith::ConstantIndexOp>(loc, 1),
                    ValueRange{finalMax}, // 作为迭代参数的初始值, 第一次迭代：args[0] 等于 finalMax
                    [&](OpBuilder &b, Location l, Value kh, ValueRange args) {      // auto iter args
                        Value currentMaxH = args[0];

                        // 遍历
                        auto loopW = b.create<scf::ForOp>(
                            l,
                            b.create<arith::ConstantIndexOp>(l, 0),
                            b.create<arith::ConstantIndexOp>(l, kernelW),
                            b.create<arith::ConstantIndexOp>(l, 1),
                            ValueRange{currentMaxH},
                            [&](OpBuilder &b2, Location l2, Value kw, ValueRange args2) {
                                Value currentMax = args2[0];

                                // h_in = h_base + kh
                                Value h_in = b2.create<arith::AddIOp>(l2, h_base, kh);
                                // w_in = w_base + kw
                                Value w_in = b2.create<arith::AddIOp>(l2, w_base, kw);

                                // 边界检查 (Boundary Check)
                                // check: h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in
                                // 这里需要获取 Input Shape 的静态值
                                Value H_in_dim = b2.create<arith::ConstantIndexOp>(l2, inputType.getDimSize(1));
                                Value W_in_dim = b2.create<arith::ConstantIndexOp>(l2, inputType.getDimSize(2));

                                // 判断逻辑 (略微繁琐，简写思路)
                                // cond = (h_in >= 0) && (h_in < H) && (w_in >= 0) && (w_in < W)
                                // 注意：Index 默认是无符号还是有符号取决于 dialect，通常用 arith.cmpi
                                Value h_ge_0 = b2.create<arith::CmpIOp>(l2, arith::CmpIPredicate::sge, h_in, b2.create<arith::ConstantIndexOp>(l2, 0));
                                Value h_lt_H = b2.create<arith::CmpIOp>(l2, arith::CmpIPredicate::slt, h_in, H_in_dim);
                                Value w_ge_0 = b2.create<arith::CmpIOp>(l2, arith::CmpIPredicate::sge, w_in, b2.create<arith::ConstantIndexOp>(l2, 0));
                                Value w_lt_W = b2.create<arith::CmpIOp>(l2, arith::CmpIPredicate::slt, w_in, W_in_dim);
                                
                                Value validH = b2.create<arith::AndIOp>(l2, h_ge_0, h_lt_H);
                                Value validW = b2.create<arith::AndIOp>(l2, w_ge_0, w_lt_W);
                                Value isValid = b2.create<arith::AndIOp>(l2, validH, validW);

                                auto ifOp = b2.create<scf::IfOp>(
                                    l2, elementType, isValid, /*withElseRegion=*/true
                                );

                                // Then Block (Valid Input)
                                {
                                    OpBuilder thenBuilder = ifOp.getThenBodyBuilder();
                                    Value val = thenBuilder.create<memref::LoadOp>(l2, input, ValueRange{n, h_in, w_in, c});
                                    thenBuilder.create<scf::YieldOp>(l2, val);
                                }

                                // Else Block (Padding) -> Return -inf
                                {
                                    OpBuilder elseBuilder = ifOp.getElseBodyBuilder();
                                    elseBuilder.create<scf::YieldOp>(l2, initVal);
                                }

                                Value loadedVal = ifOp->getResult(0);

                                // Update Max
                                Value newMax;
                                if (elementType.isF32()) 
                                    newMax = b2.create<arith::MaximumFOp>(l2, currentMax, loadedVal);
                                else 
                                    newMax = b2.create<arith::MaxSIOp>(l2, currentMax, loadedVal);

                                b2.create<scf::YieldOp>(l2, newMax);
                            }
                        );
                        b.create<scf::YieldOp>(l, loopW.getResult(0));
                    }
                );

                finalMax = loopH.getResult(0);
                Builder.create<memref::StoreOp>(loc, finalMax, output, localIvs);
                // Builder.create<scf::YieldOp>(loc);
                Builder.create<scf::ReduceOp>(loc);
            }
        );
        rewriter.replaceOp(op, output);
        return success();
    }
};


struct TransposeOpPatNew : OpConversionPattern<npu::TransposeOp> {
    using OpConversionPattern<npu::TransposeOp>::OpConversionPattern;
    LogicalResult matchAndRewrite(npu::TransposeOp op, OpAdaptor adaptor, ConversionPatternRewriter & rewriter) const override {
        Value input = adaptor.getInput();
        auto inputType = cast<MemRefType>(input.getType());

        auto loc = op->getLoc();
        Type originalResultType = op.getType();
        Type convertedType = getTypeConverter()->convertType(originalResultType);
        auto resultType = cast<MemRefType>(convertedType);

        auto output = rewriter.create<memref::AllocOp>(loc, resultType);

        SmallVector<int64_t, 4> perms;
        auto permsAttr = op.getPerms();
        for(auto p : permsAttr)
            perms.push_back(cast<IntegerAttr>(p).getInt());

        // 准备循环边界
        SmallVector<Value> lowerBounds, upperBounds, steps; // 改成 Value
        
        for (int64_t dimSize : inputType.getShape()) {
            // 使用 create<arith::ConstantIndexOp>
            lowerBounds.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
            upperBounds.push_back(rewriter.create<arith::ConstantIndexOp>(loc, dimSize));
            steps.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 1));
        }


        rewriter.create<scf::ParallelOp>(
            loc,
            lowerBounds,
            upperBounds,
            steps,
            [&](OpBuilder &Builder, Location Loc, ValueRange localIvs) {
                // ivs 是输入的坐标索引，例如 [n, h, w, c]
                
                // A. 读取输入数据, localIvs 即按顺序读取
                Value val = Builder.create<memref::LoadOp>(Loc, input, localIvs);

                // B. 计算输出坐标
                // 规则：Output 的第 k 维坐标 = Input 的第 perms[k] 维坐标
                // after translate O[a, b, c, d] = I[a, d, b, c] or I[f([a, b, c, d])]
                // f(f(a, b, c, d)) = [a, b, c, d]
                // I[a, b, c, d] = I[f(f([a, b, c, d]))] = O[f([a, b, c, d])]

                SmallVector<Value> outputIndices;
                outputIndices.reserve(perms.size());

                for(size_t i = 0; i < perms.size(); i++) {
                    outputIndices.push_back(localIvs[perms[i]]);
                }

                Builder.create<memref::StoreOp>(Loc, val, output, outputIndices);
                Builder.create<scf::ReduceOp>(loc);
            }
        );

        rewriter.replaceOp(op, output);
        return success();
    }
};

struct MatmulOpPatNew : OpConversionPattern<npu::MatmulOp> {
    using OpConversionPattern<npu::MatmulOp>::OpConversionPattern;
    LogicalResult matchAndRewrite(npu::MatmulOp op, OpAdaptor adaptor, ConversionPatternRewriter & rewriter) const override {
        Value a = adaptor.getA();
        Value b = adaptor.getB();

        Location loc = op->getLoc();

        // 1. 获取类型信息
        auto aType = cast<MemRefType>(a.getType());
        auto resultType = cast<MemRefType>(getTypeConverter()->convertType(op.getType()));
        auto elementType = resultType.getElementType();

        // 2. 分配输出内存
        Value output = rewriter.create<memref::AllocOp>(loc, resultType);

        // 3. 获取维度大小 (B, M, K, N)
        // A: [B, M, K]
        // B: [B, K, N]
        // Out: [B, M, N]
        // 这里假设是静态 Shape，如果需要动态 Shape，请用 rewriter.create<memref::DimOp>
        int64_t B = resultType.getDimSize(0);
        int64_t M = resultType.getDimSize(1);
        int64_t N = resultType.getDimSize(2);
        int64_t K = aType.getDimSize(2); // K 来自 A 的最后一维

        Value zeroVal;
        if (elementType.isF32()) {
            zeroVal = rewriter.create<arith::ConstantOp>(loc, FloatAttr::get(elementType, 0.0));
        } else if (elementType.isInteger()) {
            zeroVal = rewriter.create<arith::ConstantOp>(loc, IntegerAttr::get(elementType, 0));
        } else {
            return rewriter.notifyMatchFailure(op, "Unsupported element type for MatMul");
        }

        // 5. 构建并行循环: 遍历 [b, m, n]
        SmallVector<Value> lowerBounds, upperBounds, steps;
        
        // 创建常数 0 和 1 (复用以减少 IR 冗余)
        Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);

        // B loop
        lowerBounds.push_back(c0);
        upperBounds.push_back(rewriter.create<arith::ConstantIndexOp>(loc, B));
        steps.push_back(c1);
        
        // M loop
        lowerBounds.push_back(c0);
        upperBounds.push_back(rewriter.create<arith::ConstantIndexOp>(loc, M));
        steps.push_back(c1);

        // N loop
        lowerBounds.push_back(c0);
        upperBounds.push_back(rewriter.create<arith::ConstantIndexOp>(loc, N));
        steps.push_back(c1);

        rewriter.create<scf::ParallelOp>(
            loc,
            lowerBounds,
            upperBounds,
            steps,
            [&](OpBuilder &builder, Location loc, ValueRange ivs) {
                // ivs = [b, m, n]
                Value idx_b = ivs[0];
                Value idx_m = ivs[1];
                Value idx_n = ivs[2];

                // 6. 内层循环 K (Reduction)
                // 使用 scf::ForOp 并带上 iter_args 来做累加
                auto loopK = builder.create<scf::ForOp>(
                    loc,
                    builder.create<arith::ConstantIndexOp>(loc, 0),
                    builder.create<arith::ConstantIndexOp>(loc, K),
                    builder.create<arith::ConstantIndexOp>(loc, 1),
                    ValueRange{zeroVal},
                    [&](OpBuilder &b2, Location l2, Value idx_k, ValueRange args) {
                        Value currentSum = args[0];

                        // Load A[b, m, k]
                        Value valA = b2.create<memref::LoadOp>(l2, a, ValueRange{idx_b, idx_m, idx_k});
                        
                        // Load B[b, k, n]
                        Value valB = b2.create<memref::LoadOp>(l2, b, ValueRange{idx_b, idx_k, idx_n});

                        // Compute: sum += A * B
                        Value product;
                        Value newSum;
                        
                        if (elementType.isF32()) {
                            product = b2.create<arith::MulFOp>(l2, valA, valB);
                            newSum = b2.create<arith::AddFOp>(l2, currentSum, product);
                        } else {
                            product = b2.create<arith::MulIOp>(l2, valA, valB);
                            newSum = b2.create<arith::AddIOp>(l2, currentSum, product);
                        }

                        // Yield new sum to next iteration
                        b2.create<scf::YieldOp>(l2, newSum);
                    }
                );

                // 7. 将最终的 sum 存入 Output[b, m, n]
                Value finalSum = loopK->getResult(0);
                builder.create<memref::StoreOp>(loc, finalSum, output, ValueRange{idx_b, idx_m, idx_n});
                builder.create<scf::ReduceOp>(loc);
            }
        );
        rewriter.replaceOp(op, output);
        return success();
    }
};


struct AddOpPatNew : OpConversionPattern<npu::AddOp> {
    using OpConversionPattern<npu::AddOp>::OpConversionPattern;
    LogicalResult matchAndRewrite(npu::AddOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
        Value input1 = adaptor.getInput1();
        Value input2 = adaptor.getInput2();

        auto loc = op->getLoc();

        Type originalResultType = op.getType();
        Type convertedType = getTypeConverter()->convertType(originalResultType);
        auto resultType = cast<MemRefType>(convertedType);

        auto output = rewriter.create<memref::AllocOp>(loc, resultType);

        auto shape = resultType.getShape();
        SmallVector<Value, 4> lowerBounds(shape.size(),rewriter.create<arith::ConstantIndexOp>(loc, 0));
        SmallVector<Value, 4> steps(shape.size(), rewriter.create<arith::ConstantIndexOp>(loc, 1));

        SmallVector<Value, 4> upperBounds;
        for(int64_t dim : shape) {
            upperBounds.push_back(rewriter.create<arith::ConstantIndexOp>(loc, dim));
        }

        // TODO: not use vector, need add vector
        rewriter.create<scf::ParallelOp>(
            loc, lowerBounds, upperBounds, steps, 
            [&](OpBuilder &Builder, Location Loc, ValueRange localIvs) {
                // 在这里，你可以访问所有被捕获的变量
                // 例如：input1, input2, loc, lowerBounds, steps, upperBounds
                Value lhs = Builder.create<memref::LoadOp>(Loc, input1, localIvs);
                Value rhs = Builder.create<memref::LoadOp>(Loc, input2, localIvs);

                Value result = Builder.create<arith::AddFOp>(loc, lhs, rhs);

                Builder.create<memref::StoreOp>(loc, result, output, localIvs);
                // Builder.create<scf::YieldOp>(loc);
                Builder.create<scf::ReduceOp>(loc);

            });

        rewriter.replaceOp(op, output);
        return success();
    }
};

struct ReturnOpPatNew : public OpConversionPattern<func::ReturnOp> {
    using OpConversionPattern<func::ReturnOp>::OpConversionPattern;
  
    LogicalResult matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
      // 这里的 adaptor.getOperands() 会自动获取经过 TypeConverter 转换后的新操作数
      rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
      return success();
    }
};
  

struct ConvertNpuToVecScfPass : npu::impl::ConvertNpuToVecScfBase<ConvertNpuToVecScfPass> {
    using npu::impl::ConvertNpuToVecScfBase<ConvertNpuToVecScfPass>::ConvertNpuToVecScfBase;
    void getDependentDialects(DialectRegistry &registry) const final {
      registry.insert<NpuDialect>();
      registry.insert<func::FuncDialect>();
      registry.insert<scf::SCFDialect>();
      registry.insert<memref::MemRefDialect>();
      registry.insert<arith::ArithDialect>();
      registry.insert<vector::VectorDialect>();
    }

    void runOnOperation() final {
        ConversionTarget target(getContext());
        target.addLegalDialect<scf::SCFDialect, memref::MemRefDialect, arith::ArithDialect>();
        target.addIllegalDialect<npu::NpuDialect>();
        // 允许转换框架生成的胶水代码（Cast Op）存在
        target.addLegalOp<UnrealizedConversionCastOp>();

        TypeConverter converter;
        converter.addConversion([&](NpuTensorType type) -> Type {
            return MemRefType::get( type.getShape(), type.getElementType());
        });

        converter.addConversion([&](NpuShapeType type) -> Type {
            int rank = type.getRank();
            return MemRefType::get({rank}, IndexType::get(type.getContext()));
        });

        // 处理 Source -> Target 的转换 (例如在 Op 内部创建新类型)
        converter.addTargetMaterialization([](OpBuilder& builder, Type resultType, ValueRange inputs, Location loc) -> Value {
            return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs).getResult(0);
        });

        converter.addSourceMaterialization([](OpBuilder& builder, Type resultType, ValueRange inputs, Location loc) -> Value {
            return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs).getResult(0);
        });

        // 只有当函数的签名（Arguments/Results）符合 Converter 的规则时，FuncOp 才合法
        target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp f) {
            return llvm::all_of(f.getArgumentTypes(), 
                [](Type t) {return !isa<npu::NpuTensorType>(t);});
        });

        auto checkValid = [](Operation* f) {
            return llvm::all_of(f->getOperandTypes(), [](Type t) {return !isa<npu::NpuTensorType>(t);});
        };
        target.addDynamicallyLegalOp<func::ReturnOp>(checkValid);


        RewritePatternSet patterns(&getContext());

        // patterns.add<Conv2dOpPat, ConstantOpPat, ReturnOpPat, ConstantShapeOpPat, ReshapeOpPat, ClampOpPat, MaxPool2dOpPat,
        //     TransposeOpPat, MatmulOpPat, AddOpPat>(converter, &getContext());
        // 添加 npu 操作转换模式

        patterns.add<ConstantOpPatNew, ConstantShapeOpPatNew>(converter, &getContext());
        patterns.add<ReturnOpPatNew>(converter, &getContext());
        patterns.add<AddOpPatNew, Conv2dOpPatNew, ReshapeOpPatNew, ClampOpPatNew, MaxPool2dOpPatNew, TransposeOpPatNew, MatmulOpPatNew>(converter, &getContext());
        
        
        populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, converter);
        if(failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
            signalPassFailure();
    }
};

std::unique_ptr<mlir::Pass> npu::createConvertNpuToVecScfPass(ConvertNpuToVecScfOptions options) {
    return std::make_unique<ConvertNpuToVecScfPass>(options);
}

// std::unique_ptr<mlir::Pass> npu::createConvertTosaToNpuPass(ConvertTosaToNpuOptions options) {
//     return std::make_unique<ConvertTosaToNpuPass>(options);
// }

