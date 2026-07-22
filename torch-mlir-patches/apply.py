#!/usr/bin/env python3
"""Apply the EOT bridge to a concrete torch-mlir source checkout."""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch-mlir-source", required=True, type=Path)
    args = parser.parse_args()
    source_root = args.torch_mlir_source.resolve(strict=True)
    conversion_dir = source_root / "lib" / "Conversion" / "TorchToTosa"
    conversion_cpp = conversion_dir / "TorchToTosa.cpp"
    cmake_file = conversion_dir / "CMakeLists.txt"
    if not conversion_cpp.is_file() or not cmake_file.is_file():
        raise FileNotFoundError(
            "expected lib/Conversion/TorchToTosa/{TorchToTosa.cpp,CMakeLists.txt}")

    cpp_text = conversion_cpp.read_text(encoding="utf-8")
    cmake_text = cmake_file.read_text(encoding="utf-8")
    include_line = '#include "TorchEotCustomToTosa.h"\n'
    if include_line not in cpp_text:
        anchor = '#include "torch-mlir/Conversion/TorchToTosa/TorchToTosa.h"\n'
        if anchor not in cpp_text:
            raise RuntimeError("matching TorchToTosa include anchor was not found")
        cpp_text = cpp_text.replace(anchor, anchor + include_line, 1)

    call = "    populateTorchEotCustomToTosaPatterns(typeConverter, patterns, target);\n"
    if call not in cpp_text:
        pattern = re.compile(
            r"(\s*auto allConvertibleOps\s*=\s*"
            r"populateTorchToTosaConversionPatternsAndIllegalOps\("
            r"\s*typeConverter,\s*patterns\s*\);\n)")
        match = pattern.search(cpp_text)
        if not match:
            raise RuntimeError("matching Torch-to-TOSA pattern population anchor was not found")
        cpp_text = cpp_text[:match.end()] + call + cpp_text[match.end():]

    if "TorchEotCustomToTosa.cpp" not in cmake_text:
        match = re.search(r"^(\s*)TorchToTosa\.cpp\s*$", cmake_text, re.MULTILINE)
        if not match:
            raise RuntimeError("TorchToTosa.cpp CMake source entry was not found")
        insertion = match.group(0) + "\n" + match.group(1) + "TorchEotCustomToTosa.cpp"
        cmake_text = cmake_text[:match.start()] + insertion + cmake_text[match.end():]

    local_dir = Path(__file__).resolve().parent
    shutil.copy2(local_dir / "TorchEotCustomToTosa.cpp",
                 conversion_dir / "TorchEotCustomToTosa.cpp")
    shutil.copy2(local_dir / "TorchEotCustomToTosa.h",
                 conversion_dir / "TorchEotCustomToTosa.h")
    conversion_cpp.write_text(cpp_text, encoding="utf-8")
    cmake_file.write_text(cmake_text, encoding="utf-8")
    commit = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=False, text=True, capture_output=True).stdout.strip() or "unknown"
    print(f"applied EOT Torch-to-TOSA bridge to torch-mlir commit {commit}")


if __name__ == "__main__":
    main()
