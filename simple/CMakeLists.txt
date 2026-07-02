cmake_minimum_required(VERSION 3.16)
project(fa_simple)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "ASCEND_HOME_PATH not set. Run: source /usr/local/Ascend/ascend-toolkit/set_env.sh")
endif()
set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})

# TILE_S1 tiling granularity (compile-time). Override: cmake -DFA_TILE_S1=512
set(FA_TILE_S1 256 CACHE STRING "TILE_S1 tiling granularity")
add_compile_definitions(FA_TILE_S1=${FA_TILE_S1})

# AI Core arch for Ascend 910B1/B2/B3 (A2). Change to dav-c310 for Ascend910_9599 (A3).
set(CCE_AICORE_ARCH "dav-c220")

set(CMAKE_C_COMPILER bisheng)
set(CMAKE_CXX_COMPILER bisheng)

add_compile_options(-D_FORTIFY_SOURCE=2 -O2 -std=c++17
    -Wno-macro-redefined -Wno-ignored-attributes -fstack-protector-strong)
add_link_options(-s -Wl,-z,relro -Wl,-z,now)

include_directories(
    ${PROJECT_SOURCE_DIR}/../refs/pto-isa/include   # pto framework headers (from refs/)
    ${ASCEND_HOME_PATH}/include
    ${ASCEND_HOME_PATH}/pkg_inc
    ${ASCEND_HOME_PATH}/pkg_inc/runtime
    ${ASCEND_HOME_PATH}/pkg_inc/profiling
    ${ASCEND_HOME_PATH}/include/experiment/runtime
    ${ASCEND_HOME_PATH}/include/experiment/msprof
)

# --- Device kernel (.so), compiled by the CCE path ---
set(CCE_OPTS -xcce -Xhost-start -Xhost-end
    "SHELL:-mllvm -cce-aicore-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-function-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-record-overflow=true"
    "SHELL:-mllvm -cce-aicore-addr-transform"
    "SHELL:-mllvm -cce-aicore-dcci-insert-for-scalar=false")

add_library(fa_performance_kernel SHARED fa_performance_kernel.cpp)
target_compile_options(fa_performance_kernel PRIVATE ${CCE_OPTS} --cce-aicore-arch=${CCE_AICORE_ARCH} -std=c++17)
target_link_options(fa_performance_kernel PRIVATE --cce-fatobj-link)

# --- C-ABI launcher (.so) for the torch PoC, host C++, links the kernel .so ---
add_library(tfa_torch SHARED tfa_torch_launch.cpp)
target_compile_options(tfa_torch PRIVATE -xc++ "SHELL:-include stdint.h" "SHELL:-include stddef.h")
target_link_directories(tfa_torch PUBLIC ${ASCEND_HOME_PATH}/lib64)
target_link_libraries(tfa_torch PRIVATE
    fa_performance_kernel
    runtime stdc++ ascendcl m tiling_api platform c_sec dl nnopbase pthread)

# Non-default tile builds get distinct lib names so both variants can be dlopen'd in one process
# (same soname would make the loader reuse the first-loaded kernel, hiding the other's symbols).
if(NOT FA_TILE_S1 EQUAL 256)
    set_target_properties(fa_performance_kernel PROPERTIES OUTPUT_NAME "fa_performance_kernel_tile${FA_TILE_S1}")
    set_target_properties(tfa_torch PROPERTIES OUTPUT_NAME "tfa_torch_tile${FA_TILE_S1}")
endif()
