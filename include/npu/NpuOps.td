#ifndef NPU_OPS_TD
#define NPU_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Dialect/Tosa/IR/TosaTypesBase.td"
include "mlir/Dialect/Tosa/IR/TosaOpBase.td"

// def NPU_Tensor4D : AnyRankedTensorOf<[F32, F16]>;
// def Tosa_Tensor4D : AnyTypeOf<[Tosa_UnrankedTensor, TosaTensorRankOf<[Tosa_AnyNumber], [4]>], "4-d tosa-conformant tensor", "::mlir::TensorType">;

class NpuOp<string mnemonic, list<Trait> traits = []> :
  Op<NpuDialect, mnemonic, traits>;

def Conv2dOp : NpuOp<"conv2d"> {
    let summary = "conv2d operation";
    let arguments = (ins
      // **保留：** 输入 (TOSA_Tensor4D -> NPU_Tensor4D)
      NpuTensor:$input,
      
      // **保留：** 权重 (TOSA_Tensor4D -> NPU_Tensor4D)
      NpuTensor:$weight,
      
      // **保留：** 偏置 (TOSA_Tensor1D -> NPU_Tensor1D)
      NpuTensor:$bias, //

      // **保留：** 填充 (TOSA_IntArrayAttr4 -> I64ArrayAttr)
      // 填充是核心参数，必须保留。
      I64ArrayAttr:$pad, 
      
      // **保留：** 步长 (TOSA_IntArrayAttr2 -> I64ArrayAttr)
      // 步长是核心参数，必须保留。
      I64ArrayAttr:$stride,
      
      // **保留：** 膨胀 (TOSA_IntArrayAttr2 -> I64ArrayAttr)
      // 膨胀（Dilation）也需要保留，否则无法实现空洞卷积。
      I64ArrayAttr:$dilation
      
      // **移除：** 所有量化参数 (zp, acc_type, local_bound)
      // 假设在 TOSA -> NPU 转换前，模型已经完成了解量化。
      // 如果你要支持量化，应该将这些参数简化为 NPU 特定的格式。
    );

    let results = (outs
      // **保留：** 输出 (TOSA_Tensor4D -> NPU_Tensor4D)
      NpuTensor:$output
    );
    let assemblyFormat = [{
      `(` $input `,` $weight `,` $bias `)` 
      attr-dict `:` type($input) `,` type($weight) `,` type($bias) `->` type($output)
    }];
}

def ConstantOp : NpuOp<"const", [Pure, FirstAttrDerivedResultType]> {
  let summary = "const operation";
  let arguments = (ins ElementsAttr:$values);
  let results = (outs NpuTensor:$result);
  let assemblyFormat = "$values attr-dict `:` type($result)";
}

def ConstantShapeOp : NpuOp<"const_shape", [Pure]> {
  let summary = "Constant Shape operator.";
  // let description = [{
  //   ```mlir
  //   // Generic form
  //   %out = "tosa.const_shape"() {values = dense<0> : tensor<4xindex>} : () -> !tosa.shape<4>
  //   ```
  // }];

  let arguments = (ins IndexElementsAttr:$values);
  let results = (outs NpuShape:$result);
  let assemblyFormat = [{
    `{` `values` `=` $values `}` attr-dict `:` `(` `)` `->` type($result)
  }];
}

def ReshapeOp : NpuOp<"reshape"> {
  let summary = "reshape operation";
  let arguments = (ins
    NpuTensor:$input,
    NpuShape:$shape
  );
  let results = (outs NpuTensor:$output);
  let assemblyFormat = [{
    `(` $input `,` $shape `)` attr-dict `:` type($input) `,` type($shape) `->` type($output)
  }];
}

def ClampOp : NpuOp<"clamp"> {
  let summary = "clamp operation";
  let arguments = (ins
    NpuTensor:$input,
    F32Attr:$min_val,
    F32Attr:$max_val
  );
  let results = (outs NpuTensor:$output);
  let assemblyFormat = [{
    `(` $input `)` attr-dict `:` `(` type($input) `)` `->` type($output)
  }];
}

def MaxPool2dOp : NpuOp<"maxpool2d"> {
  let summary = "maxpool2d operation";
  let arguments = (ins
    NpuTensor:$input,
    I64ArrayAttr:$kernel,
    I64ArrayAttr:$stride,
    I64ArrayAttr:$pad
  );

  let results = (outs
    NpuTensor:$output
  );
  let assemblyFormat = [{
    `(` $input `)`
    attr-dict `:` type($input) `->` type($output)
  }];
}

def TransposeOp : NpuOp<"transpose"> {
  let summary = "transpose operation";
  let arguments = (ins
    NpuTensor:$input,
    I32ArrayAttr:$perms
  );

  let results = (outs
    NpuTensor:$output
  );

  let assemblyFormat = [{
    `(` $input `)` attr-dict `:` type($input) `->` type($output)
  }];
}

def MatmulOp : NpuOp<"matmul", [Pure]> {
  let summary = "matmul operation";
  let arguments = (ins
    NpuTensor:$a,
    NpuTensor:$b
  );

  let results = (outs
    NpuTensor:$output
  );

  let assemblyFormat = [{
    `(` $a `,` $b `)` attr-dict `:` `(` type($a) `,` type($b) `)` `->` type($output)
  }];
}

def AddOp : NpuOp<"Add", [Pure, SameOperandsAndResultType]> {
  let summary = "add operation";
  let arguments = (ins
    NpuTensor:$input1,
    NpuTensor:$input2
  );

  let results = (outs
    NpuTensor:$output
  );

  let assemblyFormat = [{
    `(` $input1 `,` $input2 `)` attr-dict `:` type($input1) `,` type($input2) `->` type($output)
  }];
}

#endif
