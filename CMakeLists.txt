cmake_minimum_required(VERSION 3.0)

project(dkm)

set(DEBUG_FLAGS "-std=c++11 -g -O1 -Wall -Wextra -Werror -Wpedantic")
set(RELEASE_FLAGS "-std=c++11 -O3 -Wall -Wextra -Werror -Wpedantic")

set(CMAKE_CXX_FLAGS ${RELEASE_FLAGS})
set(CMAKE_CXX_FLAGS_DEBUG ${DEBUG_FLAGS})
set(CMAKE_CONFIGURATION_TYPES Debug Release)

add_subdirectory(src)