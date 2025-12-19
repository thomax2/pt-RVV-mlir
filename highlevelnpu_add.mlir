module {
  // 定义一个名为 @add_example 的函数
  // 它接受一个 1x1 的 f32 张量作为输入 (%arg0)
  // 并返回一个 1x1 的 f32 张量
  func.func @add_example(%arg0: !npu.tensor<1x1x f32>) -> !npu.tensor<1x1x f32> {
    // 使用 npu.const 创建一个常量张量 %0
    // 它的值是 5.0，形状是 1x1，类型是 f32
    // %0 = npu.const dense<5.000000e+00> : 
    %0 = npu.const dense<0.000000e+00> : tensor<1x1xf32> : <1x1x f32>
    %3 = npu.const_shape{values = dense<[1, 1, 32]> : tensor<3xindex>} : () -> <3>
    // 使用 npu.Add 将输入张量 %arg0 和常量张量 %0 相加
    // 结果保存在 %1 中
    %1 = npu.Add(%arg0, %0) : <1x1x f32>, <1x1x f32> -> <1x1x f32>

    // 返回相加后的结果 %1
    return %1 : !npu.tensor<1x1x f32>
  }
}
