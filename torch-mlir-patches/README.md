# Torch-MLIR EOT bridge source extension

These two sources are the project-maintained Torch-to-TOSA extension. They are
kept separate because this checkout does not contain torch-mlir sources or a
torch-mlir package/build from which a matching commit can be identified.

Apply it to the exact checkout used by the Python package/build:

```bash
python torch-mlir-patches/apply.py --torch-mlir-source "$TORCH_MLIR_SOURCE"
```

The applicator checks the current source anchors before writing, adds the files
to `lib/Conversion/TorchToTosa/` and its existing CMake target, then inserts the
following call after the standard illegal-operation registration loop:

```cpp
populateTorchEotCustomToTosaPatterns(typeConverter, patterns, target);
```

The pattern is deliberately part of the existing dialect conversion. It uses
the pass's `TypeConverter`, converted operand adaptor, result conversion and
`ConversionTarget`; scalar operands are read from the original Torch op only
when they are compile-time constants. No unrealized casts are created.
The applicator excludes `torch.constant.float` from the standard pass's static
illegal-operation loop, then the bridge applies stricter dynamic legality:
only constants used exclusively as EOT compile-time attributes are temporarily
legal. The EOT rewrite consumes and erases them; unrelated Torch float
constants remain illegal. The bridge registration call remains after the
standard illegal-operation loop.

The source accepts the importer spellings `torch.eot.name`, `eot::name`,
`eot.name`, and their `.default` variants, emits sorted deterministic JSON,
supports the nine-result `gm_parameterize`, and marks unknown `eot` operators
illegal while leaving other custom domains untouched.
