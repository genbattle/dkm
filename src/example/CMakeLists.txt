message(STATUS "Building example")

set(target dkm_example)

set(sources
	main.cpp
)

add_executable(${target} ${sources})

if(OpenMP_CXX_FOUND)
	target_link_libraries(${target} OpenMP::OpenMP_CXX)
endif()