cmake_minimum_required(VERSION 3.0)

project(dkm_bench)

set(DEBUG_FLAGS "-std=c++11 -g -O0 -Wall -Wextra -Werror -Wpedantic")
set(RELEASE_FLAGS "-std=c++11 -O3 -Wall -Wextra -Werror -Wpedantic")

set(CMAKE_CXX_FLAGS ${RELEASE_FLAGS})
set(CMAKE_CXX_FLAGS_DEBUG ${DEBUG_FLAGS})
set(CMAKE_CONFIGURATION_TYPES Debug Release)

set(target dkm_bench)
message(STATUS "Building application ${target}")

set(sources
	bench.cpp
)

find_package( OpenCV REQUIRED )
add_executable(${target} ${sources})
target_link_libraries(${target} ${OpenCV_LIBS})
configure_file("iris.data.csv" "iris.data.csv" COPYONLY)
configure_file("USCensus1990.data.csv" "USCensus1990.data.csv" COPYONLY)
