
include "mlir/IR/AttrTypeBase.td"

class NpuType<string name, list<Trait> traits=[]>: TypeDef<NpuDialect, name, traits>;

def NpuTensor: NpuType<"NpuTensor"> {
  let mnemonic = "tensor";
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "mlir::Type":$elementType
  );

  let assemblyFormat = [{
    `<` custom<Shape>($shape) $elementType `>`
  }];
}

def NpuShape : NpuType<"NpuShape"> {
  let mnemonic = "shape";
  let parameters = (ins
    "int" : $rank
  );

  let assemblyFormat = "`<` $rank `>`";
}