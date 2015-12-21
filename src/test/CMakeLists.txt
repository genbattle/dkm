message(STATUS "Building tests")

enable_testing()

set(target dkm_tests)
message(STATUS "Building application ${target}")

set(sources
	test.cpp
)

add_executable(${target} ${sources})
add_test(test ${target})
