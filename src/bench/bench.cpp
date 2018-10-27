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
	for (const auto& c : std::get<0>(result)) {
		std::cout << "(";
		for (auto v : c) {
			std::cout << v << ",";
		}
		std::cout << "), ";
	}
	std::cout << std::endl;
}

cv::Mat load_opencv(const std::string& path) {
	std::cout << "Loading OpenCV dataset " << path << "..." << std::flush;
	std::ifstream file(path);
	cv::Mat data;
	for (auto it = std::istream_iterator<std::string>(file); it != std::istream_iterator<std::string>(); ++it) {
		auto split = split_commas(*it);
		if (split.size() != 2) { // number of values in file must match expected row size
			return cv::Mat();
		}
		cv::Vec<float, 2> values;
		for (size_t i = 0; i < 2; ++i) {
			values[i] = std::stof(split[i]);
		}
		data.push_back(values);
	}
	std::cout << "done" << std::endl;
	return data;
}

template <typename T, size_t N>
std::vector<std::array<T, N>> load_dkm(const std::string& path) {
	std::cout << "Loading dkm dataset " << path << "..." << std::flush;
	std::ifstream file(path);
	std::vector<std::array<T, N>> data;
	for (auto it = std::istream_iterator<std::string>(file); it != std::istream_iterator<std::string>(); ++it) {
		auto split = split_commas(*it);
		assert(split.size() == N); // number of values must match rows in file
		std::array<T, N> row;
		std::transform(split.begin(), split.end(), row.begin(), [](const std::string& in) -> T {
			return static_cast<T>(std::stod(in));
		});
		data.push_back(row);
	}
	std::cout << "done " << std::endl;
	return data;
}


std::chrono::duration<double> profile_opencv(const cv::Mat& data, int k) {
	std::cout << "--- Profiling OpenCV kmeans ---" << std::endl;
	std::cout << "Running kmeans";
	auto start = std::chrono::high_resolution_clock::now();
	// run the bench 10 times and take the average
	for (int i = 0; i < 10; ++i) {
		std::cout << "." << std::flush;
		cv::Mat centers, labels;
		cv::kmeans(
			data, k, labels, cv::TermCriteria(cv::TermCriteria::EPS, 0, 0.01), 1, cv::KMEANS_PP_CENTERS, centers);
		(void)labels;
	}
	auto end = std::chrono::high_resolution_clock::now();
	std::cout << "done" << std::endl;
	return (end - start) / 10.0;
}

template <typename T, size_t N>
std::chrono::duration<double> profile_dkm(const std::vector<std::array<T, N>>& data, int k) {
	std::cout << "--- Profiling dkm kmeans ---" << std::endl;
	std::cout << "Running kmeans";
	auto start = std::chrono::high_resolution_clock::now();
	// run the bench 10 times and take the average
	for (int i = 0; i < 10; ++i) {
		std::cout << "." << std::flush;
		auto result = dkm::kmeans_lloyd(data, k);
		(void)result;
	}
	auto end = std::chrono::high_resolution_clock::now();
	std::cout << "done" << std::endl;
	return (end - start) / 10.0;
}

template <typename T, size_t N>
void bench_dataset(const std::string& path, uint32_t k) {
	std::cout << "## Dataset " << path << " ##" << std::endl;

	if (N == 2) {
		auto cv_data = load_opencv(path);
		auto time_opencv = profile_opencv(cv_data, k);
		std::cout << "OpenCV: "
			  << std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(time_opencv).count() << "ms"
			  << std::endl;
	} else {
		std::cout << "No OpenCV result, OpenCV only supports 1d/2d data" <<  std::endl;
	}

	auto dkm_data = load_dkm<T, N>(path);
	auto time_dkm = profile_dkm(dkm_data, k);
	std::cout << "DKM: " << std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(time_dkm).count()
			  << "ms" << std::endl;
}

int main() {
	std::cout << "# BEGINNING PROFILING #\n" << std::endl;
	bench_dataset<float, 2>("iris.data.csv", 3);
	bench_dataset<float, 2>("s1.data.csv", 15);
	bench_dataset<float, 2>("birch3.data.csv", 100);
	bench_dataset<float, 128>("dim128.data.csv", 16);

	return 0;
}