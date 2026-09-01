cmake_minimum_required(VERSION 3.16)
project(sinkhorn_normalize CXX)

if(NOT DEFINED ASCEND_CANN_PACKAGE_PATH)
    set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/cann-9.0.0")
endif()
if(NOT DEFINED SOC_VERSION)
    set(SOC_VERSION "ascend910b4")
endif()

include(${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/tikcpp/ascendc_kernel_cmake/ascendc.cmake)
ascendc_library(sinkhorn_normalize SHARED sinkhorn_normalize.cpp)
