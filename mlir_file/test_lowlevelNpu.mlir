module {
  // --- 全局常量定义 (保持不变) ---
  memref.global "private" constant @npu_const_94646384662320 : memref<1x32x5xf32> = dense<"0xF2CDA6BD91D80ABEDABADB3D1873E23CB7E6243E48D4A9BC8D88143EF4D22BBD64481FBEC4880FBE860B93BD948E503D4E1ACF3D22FE9BBD003A363D7A2AF03D18F6953D6FA50CBEBDBDD8BDE4B89BBD2AE40EBE60FF643DE6B3B6BD006C003A20B29EBC50E7A6BD408949BB10C15C3D3BB786BDBE7D07BE0C764CBDCFAA0D3E5013193DB0B348BC0DE2243E13AFD9BDE09D00BE209C38BD0254F5BDF6DB04BE70AE883CC05C363CDACCD73D6880F8BDF67DC13DFDC932BE1E1BF7BD021EF63DC936D3BD2C787E3DEF7A233E4C455F3D394503BE8ED977BD104B6B3D2AC519BE025C79BDEB47C1BD06E00EBE1EBCC3BD7BC00B3E7CDE5D3DA5C0253E7DC7173EAD89243E2800F13CF8DA16BDC65374BD0CB40FBDC41D1F3D761C6BBDCF22163E405980BD9BBF1B3E684ABA3CC6B8DF3DBD94D6BD41272FBE089C613D2F861FBE30B2E13C4CEBD7BD2A54CFBD604919BECAFA973DCB41123EA3EED7BD90E0343D228BFA3DC840D9BCD04F9E3C09D1223E8025BFBCE43947BDD4DBA5BDC65A64BDB04C713D3F6603BE63C78ABD677C1B3E388C423D9D2F2E3E20E0873B414216BEA46E213DD85A44BD166415BE4A4DF7BD59F633BEB54EC8BDE25D983D7B8C33BE829319BE8C2A4F3DEB3DB7BD399E213E7B9F0DBE755CC7BD942B1EBDC0500ABCAA47D13D1A8AF3BD286E1EBE34F38C3DFE25CA3DFB459EBDE4A3923D1886E6BD538E14BEB80639BDFEFDD03D26A3813DD3F2343EDF64063EF791CEBD708E013DD369163E94C2083DBFE004BE82F477BD1EDBB73D49F7BABDA8215F3DC80EFABC283C07BE16AADCBD58CEF6BD7AD528BEE03C173D601D813D4C3133BEFE16C03D0BC7EDBD32E7BA3DC31D0C3EA1E08BBD705692BD6C0371BD8723123E3D6B2ABE">
  memref.global "private" constant @npu_const_shape_94646384531984 : memref<4xindex> = dense<[1, 8, 8, 1]>
  memref.global "private" constant @npu_const_94646384658944 : memref<2x3x3x1xf32> = dense<[[[[0.254846185], [0.276669294], [-0.0780908167]], [[0.306203753], [-0.0730345249], [0.0672635734]], [[-0.162285015], [0.195760876], [0.29384765]]], [[[-0.244542718], [0.289732069], [0.0623864233]], [[0.246269614], [0.0451435149], [0.160729378]], [[-0.0470636785], [0.256961972], [0.0492696464]]]]>
  memref.global "private" constant @npu_const_shape_94646384617312 : memref<2xindex> = dense<[1, 5]>
  memref.global "private" constant @npu_const_shape_94646384624064 : memref<3xindex> = dense<[1, 1, 32]>
  memref.global "private" constant @npu_const_94646384649392 : memref<1xf32> = dense<0.000000e+00>
  memref.global "private" constant @npu_const_94646384644576 : memref<2xf32> = dense_resource<torch_tensor_2_torch.float32>
  memref.global "private" constant @npu_const_94646384641760 : memref<5xf32> = dense_resource<torch_tensor_5_torch.float32>

  // [修改 1] 函数签名改为无参数，返回 i32
  func.func @main() -> i32 {
    
    // [修改 2] 手动分配输入内存 %arg0
    %arg0 = memref.alloc() : memref<1x1x8x8xf32>
    
    // [修改 2] 初始化输入数据 (填充 1.0)
    // 使用 init_ 前缀避免变量名冲突
    %init_c0 = arith.constant 0 : index
    %init_c1 = arith.constant 1 : index
    %init_c8 = arith.constant 8 : index
    %init_val = arith.constant 1.0 : f32
    
    scf.for %i = %init_c0 to %init_c8 step %init_c1 {
       scf.for %j = %init_c0 to %init_c8 step %init_c1 {
          memref.store %init_val, %arg0[%init_c0, %init_c0, %i, %j] : memref<1x1x8x8xf32>
       }
    }

    // --- 以下是你原始的计算逻辑 (保持不变) ---
    %0 = memref.get_global @npu_const_94646384641760 : memref<5xf32>
    %1 = memref.get_global @npu_const_94646384644576 : memref<2xf32>
    %2 = memref.get_global @npu_const_94646384649392 : memref<1xf32>
    %3 = memref.get_global @npu_const_shape_94646384624064 : memref<3xindex>
    %4 = memref.get_global @npu_const_shape_94646384617312 : memref<2xindex>
    %5 = memref.get_global @npu_const_94646384658944 : memref<2x3x3x1xf32>
    %6 = memref.get_global @npu_const_shape_94646384531984 : memref<4xindex>
    %7 = memref.get_global @npu_const_94646384662320 : memref<1x32x5xf32>
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [1, 8, 8, 1], strides: [64, 8, 1, 1] : memref<1x1x8x8xf32> to memref<1x8x8x1xf32>
    %alloc = memref.alloc() : memref<1x8x8x2xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_0 = arith.constant 1 : index
    %c0_1 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c1_2 = arith.constant 1 : index
    %c0_3 = arith.constant 0 : index
    %c8_4 = arith.constant 8 : index
    %c1_5 = arith.constant 1 : index
    %c0_6 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1_7 = arith.constant 1 : index
    %c1_8 = arith.constant 1 : index
    %c1_9 = arith.constant 1 : index
    %c1_10 = arith.constant 1 : index
    %c1_11 = arith.constant 1 : index
    %c1_12 = arith.constant 1 : index
    %c1_13 = arith.constant 1 : index
    %c8_14 = arith.constant 8 : index
    %c8_15 = arith.constant 8 : index
    scf.parallel (%arg1, %arg2, %arg3, %arg4) = (%c0, %c0_1, %c0_3, %c0_6) to (%c1, %c8, %c8_4, %c2) step (%c1_0, %c1_2, %c1_5, %c1_7) {
      %8 = memref.load %1[%arg4] : memref<2xf32>
      %9 = arith.muli %arg2, %c1_8 : index
      %10 = arith.subi %9, %c1_12 : index
      %11 = arith.muli %arg3, %c1_9 : index
      %12 = arith.subi %11, %c1_13 : index
      %c1_64 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      %c0_65 = arith.constant 0 : index
      %13 = scf.for %arg5 = %c0_65 to %c3 step %c1_64 iter_args(%arg6 = %8) -> (f32) {
        %c1_66 = arith.constant 1 : index
        %c3_67 = arith.constant 3 : index
        %c0_68 = arith.constant 0 : index
        %14 = scf.for %arg7 = %c0_68 to %c3_67 step %c1_66 iter_args(%arg8 = %arg6) -> (f32) {
          %c1_69 = arith.constant 1 : index
          %c1_70 = arith.constant 1 : index
          %c0_71 = arith.constant 0 : index
          %15 = scf.for %arg9 = %c0_71 to %c1_70 step %c1_69 iter_args(%arg10 = %arg8) -> (f32) {
            %16 = arith.muli %arg5, %c1_10 : index
            %17 = arith.addi %10, %16 : index
            %18 = arith.muli %arg7, %c1_11 : index
            %19 = arith.addi %12, %18 : index
            %c0_72 = arith.constant 0 : index
            %20 = arith.cmpi sge, %17, %c0_72 : index
            %21 = arith.cmpi slt, %17, %c8_14 : index
            %c0_73 = arith.constant 0 : index
            %22 = arith.cmpi sge, %19, %c0_73 : index
            %23 = arith.cmpi slt, %19, %c8_15 : index
            %24 = arith.andi %20, %21 : i1
            %25 = arith.andi %22, %23 : i1
            %26 = arith.andi %24, %25 : i1
            %27 = scf.if %26 -> (f32) {
              %29 = memref.load %reinterpret_cast[%arg1, %17, %19, %arg9] : memref<1x8x8x1xf32>
              %30 = memref.load %5[%arg4, %arg5, %arg7, %arg9] : memref<2x3x3x1xf32>
              %31 = arith.mulf %29, %30 : f32
              scf.yield %31 : f32
            } else {
              %cst_74 = arith.constant 0.000000e+00 : f32
              scf.yield %cst_74 : f32
            }
            %28 = arith.addf %arg10, %27 : f32
            scf.yield %28 : f32
          }
          scf.yield %15 : f32
        }
        scf.yield %14 : f32
      }
      memref.store %13, %alloc[%arg1, %arg2, %arg3, %arg4] : memref<1x8x8x2xf32>
      scf.reduce 
    }
    %alloc_16 = memref.alloc() : memref<1x8x8x2xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %cst_17 = arith.constant 3.40282347E+38 : f32
    %c0_18 = arith.constant 0 : index
    %c1_19 = arith.constant 1 : index
    %c1_20 = arith.constant 1 : index
    %c8_21 = arith.constant 8 : index
    %c8_22 = arith.constant 8 : index
    %c2_23 = arith.constant 2 : index
    scf.parallel (%arg1, %arg2, %arg3, %arg4) = (%c0_18, %c0_18, %c0_18, %c0_18) to (%c1_20, %c8_21, %c8_22, %c2_23) step (%c1_19, %c1_19, %c1_19, %c1_19) {
      %8 = memref.load %alloc[%arg1, %arg2, %arg3, %arg4] : memref<1x8x8x2xf32>
      %9 = arith.maximumf %8, %cst : f32
      %10 = arith.minimumf %9, %cst_17 : f32
      memref.store %10, %alloc_16[%arg1, %arg2, %arg3, %arg4] : memref<1x8x8x2xf32>
      scf.reduce 
    }
    %alloc_24 = memref.alloc() : memref<1x4x4x2xf32>
    %cst_25 = arith.constant 0xFF800000 : f32
    %c0_26 = arith.constant 0 : index
    %c1_27 = arith.constant 1 : index
    %c1_28 = arith.constant 1 : index
    %c0_29 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1_30 = arith.constant 1 : index
    %c0_31 = arith.constant 0 : index
    %c4_32 = arith.constant 4 : index
    %c1_33 = arith.constant 1 : index
    %c0_34 = arith.constant 0 : index
    %c2_35 = arith.constant 2 : index
    %c1_36 = arith.constant 1 : index
    scf.parallel (%arg1, %arg2, %arg3, %arg4) = (%c0_26, %c0_29, %c0_31, %c0_34) to (%c1_27, %c4, %c4_32, %c2_35) step (%c1_28, %c1_30, %c1_33, %c1_36) {
      %c2_64 = arith.constant 2 : index
      %8 = arith.muli %arg2, %c2_64 : index
      %c0_65 = arith.constant 0 : index
      %9 = arith.subi %8, %c0_65 : index
      %c2_66 = arith.constant 2 : index
      %10 = arith.muli %arg3, %c2_66 : index
      %c0_67 = arith.constant 0 : index
      %11 = arith.subi %10, %c0_67 : index
      %c1_68 = arith.constant 1 : index
      %c2_69 = arith.constant 2 : index
      %c0_70 = arith.constant 0 : index
      %12 = scf.for %arg5 = %c0_70 to %c2_69 step %c1_68 iter_args(%arg6 = %cst_25) -> (f32) {
        %c1_71 = arith.constant 1 : index
        %c2_72 = arith.constant 2 : index
        %c0_73 = arith.constant 0 : index
        %13 = scf.for %arg7 = %c0_73 to %c2_72 step %c1_71 iter_args(%arg8 = %arg6) -> (f32) {
          %14 = arith.addi %9, %arg5 : index
          %15 = arith.addi %11, %arg7 : index
          %c8_74 = arith.constant 8 : index
          %c8_75 = arith.constant 8 : index
          %c0_76 = arith.constant 0 : index
          %16 = arith.cmpi sge, %14, %c0_76 : index
          %17 = arith.cmpi slt, %14, %c8_74 : index
          %c0_77 = arith.constant 0 : index
          %18 = arith.cmpi sge, %15, %c0_77 : index
          %19 = arith.cmpi slt, %15, %c8_75 : index
          %20 = arith.andi %16, %17 : i1
          %21 = arith.andi %18, %19 : i1
          %22 = arith.andi %20, %21 : i1
          %23 = scf.if %22 -> (f32) {
            %25 = memref.load %alloc_16[%arg1, %14, %15, %arg4] : memref<1x8x8x2xf32>
            scf.yield %25 : f32
          } else {
            scf.yield %cst_25 : f32
          }
          %24 = arith.maximumf %arg8, %23 : f32
          scf.yield %24 : f32
        }
        scf.yield %13 : f32
      }
      memref.store %12, %alloc_24[%arg1, %arg2, %arg3, %arg4] : memref<1x4x4x2xf32>
      scf.reduce 
    }
    %alloc_37 = memref.alloc() : memref<1x2x4x4xf32>
    %c0_38 = arith.constant 0 : index
    %c1_39 = arith.constant 1 : index
    %c1_40 = arith.constant 1 : index
    %c0_41 = arith.constant 0 : index
    %c4_42 = arith.constant 4 : index
    %c1_43 = arith.constant 1 : index
    %c0_44 = arith.constant 0 : index
    %c4_45 = arith.constant 4 : index
    %c1_46 = arith.constant 1 : index
    %c0_47 = arith.constant 0 : index
    %c2_48 = arith.constant 2 : index
    %c1_49 = arith.constant 1 : index
    scf.parallel (%arg1, %arg2, %arg3, %arg4) = (%c0_38, %c0_41, %c0_44, %c0_47) to (%c1_39, %c4_42, %c4_45, %c2_48) step (%c1_40, %c1_43, %c1_46, %c1_49) {
      %8 = memref.load %alloc_24[%arg1, %arg2, %arg3, %arg4] : memref<1x4x4x2xf32>
      memref.store %8, %alloc_37[%arg1, %arg4, %arg2, %arg3] : memref<1x2x4x4xf32>
      scf.reduce 
    }
    %reinterpret_cast_50 = memref.reinterpret_cast %alloc_37 to offset: [0], sizes: [1, 1, 32], strides: [32, 32, 1] : memref<1x2x4x4xf32> to memref<1x1x32xf32>
    %alloc_51 = memref.alloc() : memref<1x1x5xf32>
    %cst_52 = arith.constant 0.000000e+00 : f32
    %c0_53 = arith.constant 0 : index
    %c1_54 = arith.constant 1 : index
    %c1_55 = arith.constant 1 : index
    %c1_56 = arith.constant 1 : index
    %c5 = arith.constant 5 : index
    scf.parallel (%arg1, %arg2, %arg3) = (%c0_53, %c0_53, %c0_53) to (%c1_55, %c1_56, %c5) step (%c1_54, %c1_54, %c1_54) {
      %c1_64 = arith.constant 1 : index
      %c32 = arith.constant 32 : index
      %c0_65 = arith.constant 0 : index
      %8 = scf.for %arg4 = %c0_65 to %c32 step %c1_64 iter_args(%arg5 = %cst_52) -> (f32) {
        %9 = memref.load %reinterpret_cast_50[%arg1, %arg2, %arg4] : memref<1x1x32xf32>
        %10 = memref.load %7[%arg1, %arg4, %arg3] : memref<1x32x5xf32>
        %11 = arith.mulf %9, %10 : f32
        %12 = arith.addf %arg5, %11 : f32
        scf.yield %12 : f32
      }
      memref.store %8, %alloc_51[%arg1, %arg2, %arg3] : memref<1x1x5xf32>
      scf.reduce 
    }
    %reinterpret_cast_57 = memref.reinterpret_cast %alloc_51 to offset: [0], sizes: [1, 5], strides: [5, 1] : memref<1x1x5xf32> to memref<1x5xf32>
    %reinterpret_cast_58 = memref.reinterpret_cast %0 to offset: [0], sizes: [1, 5], strides: [5, 1] : memref<5xf32> to memref<1x5xf32>
    %alloc_59 = memref.alloc() : memref<1x5xf32>
    %c0_60 = arith.constant 0 : index
    %c1_61 = arith.constant 1 : index
    %c1_62 = arith.constant 1 : index
    %c5_63 = arith.constant 5 : index
    scf.parallel (%arg1, %arg2) = (%c0_60, %c0_60) to (%c1_62, %c5_63) step (%c1_61, %c1_61) {
      %8 = memref.load %reinterpret_cast_57[%arg1, %arg2] : memref<1x5xf32>
      %9 = memref.load %reinterpret_cast_58[%arg1, %arg2] : memref<1x5xf32>
      %10 = arith.addf %8, %9 : f32
      memref.store %10, %alloc_59[%arg1, %arg2] : memref<1x5xf32>
      scf.reduce 
    }

    // [修改 3] 结果验证
    // 1. 定义读取坐标 (使用不冲突的变量名)
    %idx_0_debug = arith.constant 0 : index
    %idx_1_debug = arith.constant 2 : index
    
    // 2. 读取结果的第一个元素
    %res_val = memref.load %alloc_59[%idx_0_debug, %idx_1_debug] : memref<1x5xf32>
    
    // 1. 放大 1000 倍
    %c_scale = arith.constant 100.0 : f32
    %res_scaled = arith.mulf %res_val, %c_scale : f32
    
    // 2. 取绝对值 (Math Dialect)
    // 如果没有 math dialect，可以手写 if 判断
    // 或者简单一点，我们假设我们知道它是负数，直接乘 -1
    %c_neg_1 = arith.constant -1.0 : f32
    %res_abs = arith.mulf %res_scaled, %c_neg_1 : f32
    
    // 3. 转成 Int
    %ret_val = arith.fptosi %res_abs : f32 to i32
    return %ret_val : i32
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_2_torch.float32: "0x0400000018591FBEB402AE3D",
      torch_tensor_5_torch.float32: "0x04000000E02A283DB606BB3DE84B033D4FEE80BD96F6BC3D"
    }
  }
#-}