message(STATUS "Building tests")

set(target dkm_tests)
message(STATUS "Building application ${target}")

set(sources
	test.cpp
)

add_executable(${target} ${sources})
add_test(all "${EXECUTABLE_OUTPUT_PATH}/${target}")
