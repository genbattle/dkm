message(STATUS "Building benchmarks")

set(target dkm_bench)

set(sources
	bench.cpp
)

find_package(OpenCV REQUIRED)
add_executable(${target} ${sources})
target_link_libraries(${target} ${OpenCV_LIBS})
configure_file("iris.data.csv" "iris.data.csv" COPYONLY)
configure_file("USCensus1990.data.csv" "USCensus1990.data.csv" COPYONLY)
