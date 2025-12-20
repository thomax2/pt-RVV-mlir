module {
  func.func @main(%arg0: tensor<1x1x8x8xf32>) -> tensor<1x5xf32> {
    %0 = "tosa.const"() <{values = dense_resource<torch_tensor_5_torch.float32> : tensor<5xf32>}> : () -> tensor<5xf32>
    %1 = "tosa.const"() <{values = dense_resource<torch_tensor_2_torch.float32> : tensor<2xf32>}> : () -> tensor<2xf32>
    %2 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3 = tosa.const_shape  {values = dense<[1, 1, 32]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %4 = tosa.const_shape  {values = dense<[1, 5]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %5 = "tosa.const"() <{values = dense<[[[[0.254846185], [0.276669294], [-0.0780908167]], [[0.306203753], [-0.0730345249], [0.0672635734]], [[-0.162285015], [0.195760876], [0.29384765]]], [[[-0.244542718], [0.289732069], [0.0623864233]], [[0.246269614], [0.0451435149], [0.160729378]], [[-0.0470636785], [0.256961972], [0.0492696464]]]]> : tensor<2x3x3x1xf32>}> : () -> tensor<2x3x3x1xf32>
    %6 = tosa.const_shape  {values = dense<[1, 8, 8, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %7 = "tosa.const"() <{values = dense<"0xF2CDA6BD91D80ABEDABADB3D1873E23CB7E6243E48D4A9BC8D88143EF4D22BBD64481FBEC4880FBE860B93BD948E503D4E1ACF3D22FE9BBD003A363D7A2AF03D18F6953D6FA50CBEBDBDD8BDE4B89BBD2AE40EBE60FF643DE6B3B6BD006C003A20B29EBC50E7A6BD408949BB10C15C3D3BB786BDBE7D07BE0C764CBDCFAA0D3E5013193DB0B348BC0DE2243E13AFD9BDE09D00BE209C38BD0254F5BDF6DB04BE70AE883CC05C363CDACCD73D6880F8BDF67DC13DFDC932BE1E1BF7BD021EF63DC936D3BD2C787E3DEF7A233E4C455F3D394503BE8ED977BD104B6B3D2AC519BE025C79BDEB47C1BD06E00EBE1EBCC3BD7BC00B3E7CDE5D3DA5C0253E7DC7173EAD89243E2800F13CF8DA16BDC65374BD0CB40FBDC41D1F3D761C6BBDCF22163E405980BD9BBF1B3E684ABA3CC6B8DF3DBD94D6BD41272FBE089C613D2F861FBE30B2E13C4CEBD7BD2A54CFBD604919BECAFA973DCB41123EA3EED7BD90E0343D228BFA3DC840D9BCD04F9E3C09D1223E8025BFBCE43947BDD4DBA5BDC65A64BDB04C713D3F6603BE63C78ABD677C1B3E388C423D9D2F2E3E20E0873B414216BEA46E213DD85A44BD166415BE4A4DF7BD59F633BEB54EC8BDE25D983D7B8C33BE829319BE8C2A4F3DEB3DB7BD399E213E7B9F0DBE755CC7BD942B1EBDC0500ABCAA47D13D1A8AF3BD286E1EBE34F38C3DFE25CA3DFB459EBDE4A3923D1886E6BD538E14BEB80639BDFEFDD03D26A3813DD3F2343EDF64063EF791CEBD708E013DD369163E94C2083DBFE004BE82F477BD1EDBB73D49F7BABDA8215F3DC80EFABC283C07BE16AADCBD58CEF6BD7AD528BEE03C173D601D813D4C3133BEFE16C03D0BC7EDBD32E7BA3DC31D0C3EA1E08BBD705692BD6C0371BD8723123E3D6B2ABE"> : tensor<1x32x5xf32>}> : () -> tensor<1x32x5xf32>
    %8 = tosa.reshape %arg0, %6 : (tensor<1x1x8x8xf32>, !tosa.shape<4>) -> tensor<1x8x8x1xf32>
    %9 = tosa.conv2d %8, %5, %1, %2, %2 {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x8x8x1xf32>, tensor<2x3x3x1xf32>, tensor<2xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x8x8x2xf32>
    %10 = tosa.clamp %9 {max_val = 3.40282347E+38 : f32, min_val = 0.000000e+00 : f32} : (tensor<1x8x8x2xf32>) -> tensor<1x8x8x2xf32>
    %11 = tosa.max_pool2d %10 {kernel = array<i64: 2, 2>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<1x8x8x2xf32>) -> tensor<1x4x4x2xf32>
    %12 = tosa.transpose %11 {perms = array<i32: 0, 3, 1, 2>} : (tensor<1x4x4x2xf32>) -> tensor<1x2x4x4xf32>
    %13 = tosa.reshape %12, %3 : (tensor<1x2x4x4xf32>, !tosa.shape<3>) -> tensor<1x1x32xf32>
    %14 = tosa.matmul %13, %7, %2, %2 : (tensor<1x1x32xf32>, tensor<1x32x5xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x1x5xf32>
    %15 = tosa.reshape %14, %4 : (tensor<1x1x5xf32>, !tosa.shape<2>) -> tensor<1x5xf32>
    %16 = tosa.reshape %0, %4 : (tensor<5xf32>, !tosa.shape<2>) -> tensor<1x5xf32>
    %17 = tosa.add %15, %16 : (tensor<1x5xf32>, tensor<1x5xf32>) -> tensor<1x5xf32>
    return %17 : tensor<1x5xf32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_5_torch.float32: "0x04000000E02A283DB606BB3DE84B033D4FEE80BD96F6BC3D",
      torch_tensor_2_torch.float32: "0x0400000018591FBEB402AE3D"
    }
  }
#-}
