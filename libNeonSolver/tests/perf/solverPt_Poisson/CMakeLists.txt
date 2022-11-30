cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

file(GLOB_RECURSE SrcFiles src/*.cpp)

add_executable(solverPt_Poisson ${SrcFiles})

target_link_libraries(solverPt_Poisson 
	PUBLIC libNeonSolver
	PUBLIC poisson
	PUBLIC gtest_main)

set_target_properties(solverPt_Poisson PROPERTIES 
	CUDA_SEPARABLE_COMPILATION ON
	CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(solverPt_Poisson PROPERTIES FOLDER "libNeonSolver")
source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "solverPt_Poisson" FILES ${SrcFiles})

add_test(NAME solverPt_Poisson COMMAND solverPt_Poisson)


file(GLOB_RECURSE SrcFiles2 src/*.cu)

add_executable(test_shared_memory ${SrcFiles2})

target_link_libraries(test_shared_memory
	PUBLIC libNeonSolver
	PUBLIC poisson
	PUBLIC gtest_main)

set_target_properties(test_shared_memory PROPERTIES 
	CUDA_SEPARABLE_COMPILATION ON
	CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(test_shared_memory PROPERTIES FOLDER "libNeonSolver")
source_group(TREE ${CMAKE_CURRENT_LIST_DIR} PREFIX "test_shared_memory" FILES ${SrcFiles})

add_test(NAME test_shared_memory COMMAND test_shared_memory)