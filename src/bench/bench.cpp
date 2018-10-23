/*
Test cases for dkm.hpp

This is just simple test harness without any external dependencies.
*/

#include "../../include/dkm.hpp"
#include "../../include/dkm_parallel.hpp"
#include "opencv2/opencv.hpp"

#include <vector>
#include <array>
#include <tuple>
#include <string>
#include <iterator>
#include <fstream>
#include <iostream>
#include <chrono>
#include <numeric>
#include <regex>

// Split a line on commas, making it simple to pull out the values we need
std::vector<std::string> split_commas(const std::string& line) {
	std::vector<std::string> split;
	std::regex reg(",");
	std::copy(std::sregex_token_iterator(line.begin(), line.end(), reg, -1),
		std::sregex_token_iterator(),
		std::back_inserter(split));
	return split;
}

template <typename T, size_t N>
void print_result_dkm(std::tuple<std::vector<std::array<T, N>>, std::vector<uint32_t>>& result) {
	std::cout << "centers: ";
	for (auto& c : std::get<0>(result)) {
		std::cout << "(";
		for (auto v : c) {
			std::cout << v << ",";
		}
		std::cout << "), ";
	}
	std::cout << std::endl;
}

cv::Mat load_opencv() {
	std::cout << "Loading small OpenCV dataset...";
	std::ifstream file("iris.data.csv");
	cv::Mat data;
	for (auto it = std::istream_iterator<std::string>(file); it != std::istream_iterator<std::string>(); ++it) {
		auto split = split_commas(*it);
		assert(split.size() > 1); // must be at least 2 values per line
		cv::Vec<float, 2> values;
		values[0] = std::stof(split[0]);
		values[1] = std::stof(split[1]);
		data.push_back(values);
	}
	std::cout << "done" << std::endl;
	return data;
}

std::vector<std::array<float, 2>> load_dkm() {
	std::cout << "Loading small dkm dataset...";
	std::ifstream file("iris.data.csv");
	std::vector<std::array<float, 2>> data;
	for (auto it = std::istream_iterator<std::string>(file); it != std::istream_iterator<std::string>(); ++it) {
		auto split = split_commas(*it);
		assert(split.size() > 1); // must be at least 2 values per line
		data.push_back({{std::stof(split[0]), std::stof(split[1])}});
	}
	std::cout << "done" << std::endl;
	return data;
}

std::chrono::duration<double> profile_opencv(const cv::Mat& data, int k) {
	std::cout << "--- Profiling OpenCV kmeans ---" << std::endl;
	std::cout << "Running kmeans..." << std::endl;
	auto start = std::chrono::high_resolution_clock::now();
	// run the bench 10 times and take the average
	for (int i = 0; i < 10; ++i) {
		cv::Mat centers, labels;
		cv::kmeans(
			data, k, labels, cv::TermCriteria(cv::TermCriteria::EPS, 0, 0.01), 1, cv::KMEANS_PP_CENTERS, centers);
		(void)labels;
		std::cout << "centers: ";
		std::cout << centers << std::endl;
	}
	auto end = std::chrono::high_resolution_clock::now();
	std::cout << std::endl;
	std::cout << "done" << std::endl;
	return (end - start) / 10.0;
}

std::chrono::duration<double> profile_dkm(const std::vector<std::array<float, 2>>& data, int k) {
	std::cout << "--- Profiling dkm kmeans ---" << std::endl;
	std::cout << "Running kmeans..." << std::endl;
	auto start = std::chrono::high_resolution_clock::now();
	// run the bench 10 times and take the average
	for (int i = 0; i < 10; ++i) {
		auto result = dkm::kmeans_lloyd(data, k);
		print_result_dkm(result);
	}
	auto end = std::chrono::high_resolution_clock::now();
	return (end - start) / 10.0;
}

std::chrono::duration<double> profile_dkm_parallel(const std::vector<std::array<float, 2>>& data, int k) {
	std::cout << "--- Profiling dkm parallel kmeans ---" << std::endl;
	std::cout << "Running kmeans..." << std::endl;
	auto start = std::chrono::high_resolution_clock::now();
	// run the bench 10 times and take the average
	for (int i = 0; i < 10; ++i) {
		auto result = dkm_parallel::kmeans_lloyd(data, k);
		print_result_dkm(result);
	}
	auto end = std::chrono::high_resolution_clock::now();
	return (end - start) / 10.0;
}

int main() {
	std::cout << "# BEGINNING PROFILING #\n" << std::endl;
	auto cv_data = load_opencv();
	auto time_opencv = profile_opencv(cv_data, 3);
	auto dkm_data = load_dkm();
	auto time_dkm = profile_dkm(dkm_data, 3);
	auto time_dkm_parallel = profile_dkm_parallel(dkm_data, 3);

	std::cout << "OpenCV: "
			  << std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(time_opencv).count() << "ms"
			  << std::endl;
	std::cout << "DKM: " << std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(time_dkm).count()
			  << "ms" << std::endl;
	std::cout << "DKM parallel: " << std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(time_dkm_parallel).count()
			  << "ms" << std::endl;

	return 0;
}