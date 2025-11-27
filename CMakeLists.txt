cmake_minimum_required(VERSION 3.24)
project(METAL
    VERSION 1.0
    DESCRIPTION "Metal benchmark"
    LANGUAGES C CXX OBJC)

set(CMAKE_C_STANDARD 11)
set(CMAKE_OBJC_STANDARD 11)
set(CMAKE_OBJC_STANDARD_REQUIRED ON)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
set(CMAKE_OBJC_FLAGS "${CMAKE_OBJC_FLAGS} -O3")

set(OpenMP_ROOT "/opt/homebrew/opt/libomp")
find_package(OpenMP REQUIRED)

find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(METAL_FRAMEWORK      Metal      REQUIRED)
find_library(IOKIT_FRAMEWORK      IOKit      REQUIRED)
find_library(MPS_FRAMEWORK        MetalPerformanceShaders REQUIRED)

set(METAL_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul.metal
)
set(METAL_LIB default.metallib)

include_directories(BEFORE include ${CMAKE_CURRENT_SOURCE_DIR})
add_custom_command(
    OUTPUT  ${CMAKE_CURRENT_BINARY_DIR}/${METAL_LIB}
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/"
    COMMAND xcrun -sdk macosx metal -g "-I${CMAKE_CURRENT_SOURCE_DIR}" -c "${CMAKE_CURRENT_SOURCE_DIR}/matmul.metal" -o "${CMAKE_CURRENT_BINARY_DIR}/matmul.air"
    COMMAND xcrun -sdk macosx metallib "${CMAKE_CURRENT_BINARY_DIR}/matmul.air" -o "${CMAKE_CURRENT_BINARY_DIR}/${METAL_LIB}"
    DEPENDS ${METAL_SOURCES}
    COMMENT "Compiling Metal compute library"
)

add_custom_target(build_metallib ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${METAL_LIB})

add_library(metal STATIC ${CMAKE_CURRENT_SOURCE_DIR}/metal.m)
target_link_libraries(metal PRIVATE ${FOUNDATION_FRAMEWORK} ${METAL_FRAMEWORK} ${IOKIT_FRAMEWORK} ${MPS_FRAMEWORK})
    
add_executable(matmul ${CMAKE_CURRENT_SOURCE_DIR}/kernel.c ${CMAKE_CURRENT_SOURCE_DIR}/main.c ${CMAKE_CURRENT_SOURCE_DIR}/matmul.c)
target_link_libraries(matmul PRIVATE metal ${MPS_FRAMEWORK})

# 如果找到OpenMP，添加编译选项和链接库
if(OpenMP_FOUND)
    target_link_libraries(matmul PRIVATE OpenMP::OpenMP_C)
    message(STATUS "OpenMP found and linked")
else()
    message(WARNING "OpenMP not found")
endif()