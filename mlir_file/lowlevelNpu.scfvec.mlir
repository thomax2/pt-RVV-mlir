module {
  memref.global "private" constant @npu_const_94731193368112 : memref<1x32x5xf32> = dense<"0x809EAE3AD85EEBBD7AE7C6BD1ED3E1BDA7A613BE4DB9F1BDE8D4EDBCF5D11A3E575C0D3E5284A33DBDE82BBE6C093BBD2B05CFBD7E839C3D12BFDBBDD67F22BE58F787BC40FC383DA49C803DB6CCB33D1C60483D7AA7D53D803CA7BC45E3F6BD786FFA3CFE45C73D009253BBDB2716BEAC7D92BD87870CBE3A68E13D4CA0113D5871FE3CB0B108BE219BABBDEBF3AFBD094820BE78CD223DE26BB6BDC81107BEA93528BE3BAA073E58D780BC8219D3BDFE2E1FBE2174243EE0ABE9BBC0F099BC2DF0C1BD986E4EBD244A26BE583129BD01F714BE90FBCF3C9B0A25BEAA2EEABDC5B497BD36D8D8BD2083103DB614C33D8066A83B40C61F3CEF80BABDF2D5F33DBCECE8BDA8B6883CCC45033DD1DD22BE5D73013E6B9D133E67B6AEBD53D9323E746A223D0370333E4887DDBC8D7AF2BD7A3F65BDA85C01BE83F11F3E800209BE27C2CFBD18FBC6BC2031D03C29EE133E5D00043ED201FFBDE8D7603DFC8B293D3C9A0FBE70A581BDE045413DEC25FCBDD4223D3D5E11A73D81A4BFBD60A29E3B5E42ED3D9D2B86BD763D30BE21408CBD9A476FBDC07ACB3BF14231BEFBEDA9BDA16909BEBA33C33D08F7553D743496BD1AC6C63D08710BBE54E6473D6059E03C1CBE323DA0E08A3BB6959E3D4B1BE5BD062CD63D55F30B3E785F98BD202B6C3C04B8933D666F7FBD1591003EC2F5DABD4B3C2C3E004A65BBB08CEB3C43F313BE47D6053E5886CC3C138093BDE668FD3DC49C43BD624531BEF019E4BDE9911FBEE2E300BE4AF1DD3DB5012D3EE7C1163E16300ABED87FE13CFA7022BED44714BE7EB1A83D6896DB3C9AE804BE6F8FFABD3EB6E6BD42EABF3DDE03D83DE2FFB93D902C19BE0228A53DDC57133D2626C8BD2040303C072D19BEDAAF03BEFEDF953D">
  memref.global "private" constant @npu_const_shape_94731193238928 : memref<4xindex> = dense<[1, 8, 8, 1]>
  memref.global "private" constant @npu_const_94731193363584 : memref<2x3x3x1xf32> = dense<[[[[-0.0434637964], [-0.259585142], [0.0904722511]], [[0.0643031597], [0.121741802], [-0.12441501]], [[-0.18793647], [-0.0488690436], [0.208099276]]], [[[-0.21367678], [-2.47746706E-4], [0.0575312078]], [[0.299190253], [-0.126253173], [0.0712993443]], [[-0.133504912], [-0.0895605981], [0.0731403827]]]]>
  memref.global "private" constant @npu_const_shape_94731193323104 : memref<2xindex> = dense<[1, 5]>
  memref.global "private" constant @npu_const_shape_94731193331008 : memref<3xindex> = dense<[1, 1, 32]>
  memref.global "private" constant @npu_const_94731193356336 : memref<1xf32> = dense<0.000000e+00>
  memref.global "private" constant @npu_const_94731193352672 : memref<2xf32> = dense_resource<torch_tensor_2_torch.float32>
  memref.global "private" constant @npu_const_94731193348704 : memref<5xf32> = dense_resource<torch_tensor_5_torch.float32>
  func.func @main(%arg0: memref<1x1x8x8xf32>) -> memref<1x5xf32> {
    %0 = memref.get_global @npu_const_94731193348704 : memref<5xf32>
    %1 = memref.get_global @npu_const_94731193352672 : memref<2xf32>
    %2 = memref.get_global @npu_const_94731193356336 : memref<1xf32>
    %3 = memref.get_global @npu_const_shape_94731193331008 : memref<3xindex>
    %4 = memref.get_global @npu_const_shape_94731193323104 : memref<2xindex>
    %5 = memref.get_global @npu_const_94731193363584 : memref<2x3x3x1xf32>
    %6 = memref.get_global @npu_const_shape_94731193238928 : memref<4xindex>
    %7 = memref.get_global @npu_const_94731193368112 : memref<1x32x5xf32>
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
    scf.parallel (%arg1, %arg2, %arg3, %arg4) = (%c0, %c0_1, %c0_3, %c0_6) to (%c1, %c8, %c8_4, %c2) step (%c1_0, %c1_2, %c1_5, %c1_7) {
      %8 = memref.load %1[%arg4] : memref<2xf32>
      %9 = arith.muli %arg2, %c1_8 : index
      %10 = arith.subi %9, %c1_12 : index
      %11 = arith.muli %arg3, %c1_9 : index
      %12 = arith.subi %11, %c1_13 : index
      %c1_62 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      %c0_63 = arith.constant 0 : index
      %13 = scf.for %arg5 = %c0_63 to %c3 step %c1_62 iter_args(%arg6 = %8) -> (f32) {
        %c1_64 = arith.constant 1 : index
        %c3_65 = arith.constant 3 : index
        %c0_66 = arith.constant 0 : index
        %14 = scf.for %arg7 = %c0_66 to %c3_65 step %c1_64 iter_args(%arg8 = %arg6) -> (f32) {
          %c1_67 = arith.constant 1 : index
          %c1_68 = arith.constant 1 : index
          %c0_69 = arith.constant 0 : index
          %15 = scf.for %arg9 = %c0_69 to %c1_68 step %c1_67 iter_args(%arg10 = %arg8) -> (f32) {
            %16 = arith.muli %arg5, %c1_10 : index
            %17 = arith.addi %10, %16 : index
            %18 = arith.muli %arg7, %c1_11 : index
            %19 = arith.addi %12, %18 : index
            %c8_70 = arith.constant 8 : index
            %c8_71 = arith.constant 8 : index
            %c0_72 = arith.constant 0 : index
            %20 = arith.cmpi sge, %17, %c0_72 : index
            %21 = arith.cmpi slt, %17, %c8_70 : index
            %c0_73 = arith.constant 0 : index
            %22 = arith.cmpi sge, %19, %c0_73 : index
            %23 = arith.cmpi slt, %19, %c8_71 : index
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
    %alloc_14 = memref.alloc() : memref<1x8x8x2xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %cst_15 = arith.constant 3.40282347E+38 : f32
    %c0_16 = arith.constant 0 : index
    %c1_17 = arith.constant 1 : index
    %c1_18 = arith.constant 1 : index
    %c8_19 = arith.constant 8 : index
    %c8_20 = arith.constant 8 : index
    %c2_21 = arith.constant 2 : index
    scf.parallel (%arg1, %arg2, %arg3, %arg4) = (%c0_16, %c0_16, %c0_16, %c0_16) to (%c1_18, %c8_19, %c8_20, %c2_21) step (%c1_17, %c1_17, %c1_17, %c1_17) {
      %8 = memref.load %alloc[%arg1, %arg2, %arg3, %arg4] : memref<1x8x8x2xf32>
      %9 = arith.maximumf %8, %cst : f32
      %10 = arith.minimumf %9, %cst_15 : f32
      memref.store %10, %alloc_14[%arg1, %arg2, %arg3, %arg4] : memref<1x8x8x2xf32>
      scf.reduce 
    }
    %alloc_22 = memref.alloc() : memref<1x4x4x2xf32>
    %cst_23 = arith.constant 0xFF800000 : f32
    %c0_24 = arith.constant 0 : index
    %c1_25 = arith.constant 1 : index
    %c1_26 = arith.constant 1 : index
    %c0_27 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1_28 = arith.constant 1 : index
    %c0_29 = arith.constant 0 : index
    %c4_30 = arith.constant 4 : index
    %c1_31 = arith.constant 1 : index
    %c0_32 = arith.constant 0 : index
    %c2_33 = arith.constant 2 : index
    %c1_34 = arith.constant 1 : index
    scf.parallel (%arg1, %arg2, %arg3, %arg4) = (%c0_24, %c0_27, %c0_29, %c0_32) to (%c1_25, %c4, %c4_30, %c2_33) step (%c1_26, %c1_28, %c1_31, %c1_34) {
      %c2_62 = arith.constant 2 : index
      %8 = arith.muli %arg2, %c2_62 : index
      %c0_63 = arith.constant 0 : index
      %9 = arith.subi %8, %c0_63 : index
      %c2_64 = arith.constant 2 : index
      %10 = arith.muli %arg3, %c2_64 : index
      %c0_65 = arith.constant 0 : index
      %11 = arith.subi %10, %c0_65 : index
      %c1_66 = arith.constant 1 : index
      %c2_67 = arith.constant 2 : index
      %c0_68 = arith.constant 0 : index
      %12 = scf.for %arg5 = %c0_68 to %c2_67 step %c1_66 iter_args(%arg6 = %cst_23) -> (f32) {
        %c1_69 = arith.constant 1 : index
        %c2_70 = arith.constant 2 : index
        %c0_71 = arith.constant 0 : index
        %13 = scf.for %arg7 = %c0_71 to %c2_70 step %c1_69 iter_args(%arg8 = %arg6) -> (f32) {
          %14 = arith.addi %9, %arg5 : index
          %15 = arith.addi %11, %arg7 : index
          %c8_72 = arith.constant 8 : index
          %c8_73 = arith.constant 8 : index
          %c0_74 = arith.constant 0 : index
          %16 = arith.cmpi sge, %14, %c0_74 : index
          %17 = arith.cmpi slt, %14, %c8_72 : index
          %c0_75 = arith.constant 0 : index
          %18 = arith.cmpi sge, %15, %c0_75 : index
          %19 = arith.cmpi slt, %15, %c8_73 : index
          %20 = arith.andi %16, %17 : i1
          %21 = arith.andi %18, %19 : i1
          %22 = arith.andi %20, %21 : i1
          %23 = scf.if %22 -> (f32) {
            %25 = memref.load %alloc_14[%arg1, %14, %15, %arg4] : memref<1x8x8x2xf32>
            scf.yield %25 : f32
          } else {
            scf.yield %cst_23 : f32
          }
          %24 = arith.maximumf %arg8, %23 : f32
          scf.yield %24 : f32
        }
        scf.yield %13 : f32
      }
      memref.store %12, %alloc_22[%arg1, %arg2, %arg3, %arg4] : memref<1x4x4x2xf32>
      scf.reduce 
    }
    %alloc_35 = memref.alloc() : memref<1x2x4x4xf32>
    %c0_36 = arith.constant 0 : index
    %c1_37 = arith.constant 1 : index
    %c1_38 = arith.constant 1 : index
    %c0_39 = arith.constant 0 : index
    %c4_40 = arith.constant 4 : index
    %c1_41 = arith.constant 1 : index
    %c0_42 = arith.constant 0 : index
    %c4_43 = arith.constant 4 : index
    %c1_44 = arith.constant 1 : index
    %c0_45 = arith.constant 0 : index
    %c2_46 = arith.constant 2 : index
    %c1_47 = arith.constant 1 : index
    scf.parallel (%arg1, %arg2, %arg3, %arg4) = (%c0_36, %c0_39, %c0_42, %c0_45) to (%c1_37, %c4_40, %c4_43, %c2_46) step (%c1_38, %c1_41, %c1_44, %c1_47) {
      %8 = memref.load %alloc_22[%arg1, %arg2, %arg3, %arg4] : memref<1x4x4x2xf32>
      memref.store %8, %alloc_35[%arg1, %arg4, %arg2, %arg3] : memref<1x2x4x4xf32>
      scf.reduce 
    }
    %reinterpret_cast_48 = memref.reinterpret_cast %alloc_35 to offset: [0], sizes: [1, 1, 32], strides: [32, 32, 1] : memref<1x2x4x4xf32> to memref<1x1x32xf32>
    %alloc_49 = memref.alloc() : memref<1x1x5xf32>
    %cst_50 = arith.constant 0.000000e+00 : f32
    %c0_51 = arith.constant 0 : index
    %c1_52 = arith.constant 1 : index
    %c1_53 = arith.constant 1 : index
    %c1_54 = arith.constant 1 : index
    %c5 = arith.constant 5 : index
    scf.parallel (%arg1, %arg2, %arg3) = (%c0_51, %c0_51, %c0_51) to (%c1_53, %c1_54, %c5) step (%c1_52, %c1_52, %c1_52) {
      %c1_62 = arith.constant 1 : index
      %c32 = arith.constant 32 : index
      %c0_63 = arith.constant 0 : index
      %8 = scf.for %arg4 = %c0_63 to %c32 step %c1_62 iter_args(%arg5 = %cst_50) -> (f32) {
        %9 = memref.load %reinterpret_cast_48[%arg1, %arg2, %arg4] : memref<1x1x32xf32>
        %10 = memref.load %7[%arg1, %arg4, %arg3] : memref<1x32x5xf32>
        %11 = arith.mulf %9, %10 : f32
        %12 = arith.addf %arg5, %11 : f32
        scf.yield %12 : f32
      }
      memref.store %8, %alloc_49[%arg1, %arg2, %arg3] : memref<1x1x5xf32>
      scf.reduce 
    }
    %reinterpret_cast_55 = memref.reinterpret_cast %alloc_49 to offset: [0], sizes: [1, 5], strides: [5, 1] : memref<1x1x5xf32> to memref<1x5xf32>
    %reinterpret_cast_56 = memref.reinterpret_cast %0 to offset: [0], sizes: [1, 5], strides: [5, 1] : memref<5xf32> to memref<1x5xf32>
    %alloc_57 = memref.alloc() : memref<1x5xf32>
    %c0_58 = arith.constant 0 : index
    %c1_59 = arith.constant 1 : index
    %c1_60 = arith.constant 1 : index
    %c5_61 = arith.constant 5 : index
    scf.parallel (%arg1, %arg2) = (%c0_58, %c0_58) to (%c1_60, %c5_61) step (%c1_59, %c1_59) {
      %8 = memref.load %reinterpret_cast_55[%arg1, %arg2] : memref<1x5xf32>
      %9 = memref.load %reinterpret_cast_56[%arg1, %arg2] : memref<1x5xf32>
      %10 = arith.addf %8, %9 : f32
      memref.store %10, %alloc_57[%arg1, %arg2] : memref<1x5xf32>
      scf.reduce 
    }
    return %alloc_57 : memref<1x5xf32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_2_torch.float32: "0x04000000C0E5FE3B6CEC95BE",
      torch_tensor_5_torch.float32: "0x04000000586E06BE4E7FF03D9DBEB4BDC87CF23C792A043E"
    }
  }
#-}

