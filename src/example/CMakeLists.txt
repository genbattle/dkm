message(STATUS "Building example")

set(target dkm_example)

set(sources
	main.cpp
)

add_executable(${target} ${sources})