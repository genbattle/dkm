cmake_minimum_required(VERSION 3.0)

project(dkm_tests)

set(DEBUG_FLAGS "-std=c++11 -g -O0 -Wall -Wextra -Werror -Wpedantic")
set(RELEASE_FLAGS "-std=c++11 -O3 -Wall -Wextra -Werror -Wpedantic")

set(CMAKE_CXX_FLAGS ${RELEASE_FLAGS})
set(CMAKE_CXX_FLAGS_DEBUG ${DEBUG_FLAGS})
set(CMAKE_CONFIGURATION_TYPES Debug Release)

set(target dkm_tests)
message(STATUS "Building application ${target}")

include_directories(
	"../include"
)

set(sources
	test.cpp
)

add_executable(${target} ${sources})
