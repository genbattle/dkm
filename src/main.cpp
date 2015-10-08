/*
Test cases for dkm.hpp

This is just simple test harness without any external dependencies.
*/

#include "dkm.hpp"

#include <cassert>
#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <tuple>
#include <algorithm>

void check_result(bool result, std::string test_name) {
	if (result) {
		std::cout << "TEST(" << test_name << ") RESULT: PASS" << std::endl;
	} else {
		std::cout << "TEST(" << test_name << ") RESULT: FAIL" << std::endl;
	}
	assert(result);
}

template <typename T, size_t N>
void print_means(std::vector<std::array<T, N>>& values) {
	std::cout << "test_simple_2d: means(";
	for (auto mean : values) {
		std::cout << "(";
		for (auto v : mean) {
			std::cout << v << ",";
		}
		std::cout << "), ";
	}
	std::cout << std::endl;
}

bool test_simple_2d() {
	std::vector<std::array<float, 2>> data{{1.f, 1.f}, {2.f, 2.f}, {1200.f, 1200.f}, {2.f, 2.f}};
	auto result = dkm::kmeans_lloyd(data, 3);
	print_means<float, 2>(std::get<0>(result));
	// TODO: verify result
	return true;
}

int main() {
	check_result(test_simple_2d(), "test_simple_2d");
}