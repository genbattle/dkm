cmake_minimum_required(VERSION 3.5..4.0)

project(dkm)

set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable this flag to treat warnings as errors
set(DKM_WARNINGS_AS_ERRORS OFF CACHE BOOL "Treat warnings as errors")

# Enable additional warnings for all compilers
if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  # MSVC equivalent flags
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
  if (DKM_WARNINGS_AS_ERRORS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /WX")
  endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  # GCC/Clang equivalent flags
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
  if (DKM_WARNINGS_AS_ERRORS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
  endif()
endif()

find_package(OpenMP)

set(CMAKE_CONFIGURATION_TYPES Debug Release)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR})

enable_testing()

add_subdirectory(src)