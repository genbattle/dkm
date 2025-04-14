message(STATUS "Building tests")

set(target dkm_tests)
message(STATUS "Building application ${target}")

set(sources
	test.cpp
)

add_executable(${target} ${sources})
add_test(all "${EXECUTABLE_OUTPUT_PATH}/${target}")

if(OpenMP_CXX_FOUND)
	target_link_libraries(${target} OpenMP::OpenMP_CXX)
endif()

file(COPY "iris.data.csv" DESTINATION "${EXECUTABLE_OUTPUT_PATH}")