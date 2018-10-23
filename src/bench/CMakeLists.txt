message(STATUS "Building benchmarks")

set(target dkm_bench)

set(sources
	bench.cpp
)

find_package(OpenCV REQUIRED)
add_executable(${target} ${sources})
target_link_libraries(${target} ${OpenCV_LIBS} pthread)

file(COPY "iris.data.csv" DESTINATION "${EXECUTABLE_OUTPUT_PATH}")