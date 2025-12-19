set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
add_subdirectory(include/npu)
add_subdirectory(lib)
add_subdirectory(tools/npu-opt)