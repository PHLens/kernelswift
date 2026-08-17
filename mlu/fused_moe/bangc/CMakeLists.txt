cmake_minimum_required(VERSION 3.5)
project(fused_moe_bangc)

set(CMAKE_SYSTEM_NAME Linux)
set(NEUWARE_HOME $ENV{NEUWARE_HOME})
if(NOT NEUWARE_HOME)
  set(NEUWARE_HOME /projs/framework/lipenghui/neuware_home)
endif()
message(STATUS "NEUWARE_HOME=${NEUWARE_HOME}")

set(CMAKE_MODULE_PATH
  "${NEUWARE_HOME}/cmake"
  "${NEUWARE_HOME}/cmake/modules"
)
find_package(BANG REQUIRED)

set(TARGET_MLU_ARCH "compute_50" CACHE STRING "MLU arch")
set(TARGET_CPU_ARCH "x86_64-linux-gnu" CACHE STRING "CPU arch")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fPIC -std=gnu++11 -fno-strict-aliasing -pthread -O2")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -fPIC -pthread -O2")

list(APPEND BANG_CNCC_FLAGS --bang-arch=${TARGET_MLU_ARCH})
list(APPEND BANG_CNCC_FLAGS --bang-mlu-arch=mtp_592)
list(APPEND BANG_CNCC_FLAGS -Wall -fPIC -std=c++11 -O2)

link_directories("${NEUWARE_HOME}/lib64")
link_directories("${NEUWARE_HOME}/lib")

# host code needs cnrt.h (NEUWARE_HOME/include) for the device runtime API,
# and the clang-bundled bang host headers (NEUWARE_HOME/lib/clang/11.1.0/include)
# for the __half type. The clang dir is NOT added to the global include path
# because its stdint.h/stddef.h wrappers use `__has_feature` (clang-only) and
# would shadow the system versions pulled in by cnrt.h. Instead, host_driver.cpp
# uses a macro BANG_HEADER_DIR configured below to include bang_fp16.h by
# absolute path; bang_host_functions_decls.h then finds its siblings via
# quoted-relative resolution.
include_directories("${NEUWARE_HOME}/include")

bang_add_executable(fused_moe_bangc
  fused_moe_kernel.mlu
  host_driver.cpp
)

target_link_libraries(fused_moe_bangc pthread stdc++ m cnrt)

bang_add_executable(matmul_probe
  matmul_probe.mlu
  matmul_probe_host.cpp
)

target_link_libraries(matmul_probe pthread stdc++ m cnrt)
