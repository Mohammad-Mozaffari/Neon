cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

file(GLOB_RECURSE SrcFiles include/*.* src/*.*)

add_executable(fusion_stencil_map ${SrcFiles})

target_link_libraries(fusion_stencil_map
        PUBLIC libNeonSkeleton
        PUBLIC gtest_main)

target_include_directories(fusion_stencil_map PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include/")

set_target_properties(fusion_stencil_map PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(fusion_stencil_map PROPERTIES FOLDER "optimization")
source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "fusion_stencil_map" FILES ${SrcFiles})

add_test(NAME fusion_stencil_map COMMAND fusion_stencil_map)