module {
  memref.global "private" constant @npu_const_94365145164784 : memref<1x1xf32> = dense<0.000000e+00>
  func.func @add_example(%arg0: memref<1x1xf32>) -> memref<1x1xf32> {
    %0 = memref.get_global @npu_const_94365145164784 : memref<1x1xf32>
    %alloc = memref.alloc() : memref<1x1xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_0 = arith.constant 1 : index
    %c1_1 = arith.constant 1 : index
    scf.parallel (%arg1, %arg2) = (%c0, %c0) to (%c1_0, %c1_1) step (%c1, %c1) {
      %1 = memref.load %arg0[%arg1, %arg2] : memref<1x1xf32>
      %2 = memref.load %0[%arg1, %arg2] : memref<1x1xf32>
      %3 = arith.addf %1, %2 : f32
      memref.store %3, %alloc[%arg1, %arg2] : memref<1x1xf32>
      scf.reduce 
    }
    return %alloc : memref<1x1xf32>
  }
}

