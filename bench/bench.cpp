/*
Test cases for dkm.hpp

This is just simple test harness without any external dependencies.
*/

#include "../include/dkm.hpp"
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

cv::Mat load_opencv() {
	std::ifstream file("iris.data.csv");
	std::istream_iterator<std::string> input(file);
	cv::Mat data;
	for (auto it = std::istream_iterator<std::string>(file); it != std::istream_iterator<std::string>(); ++it) {
		auto comma_pos = it->find(",");
		if (comma_pos != std::string::npos) {
			cv::Vec<float, 2> values;
			values[0] = std::stof(it->substr(0, comma_pos));
			values[1] = std::stof(it->substr(comma_pos + 1));
			data.push_back(values);
		}
	}
	// std::copy(std::istream_iterator<float>(file), std::istream_iterator<float>(), data.begin<float>());
	return data;
}

std::chrono::duration<double> profile_opencv(cv::Mat& data, int k) {
	std::cout << "--- Profiling OpenCV kmeans ---" << std::endl;
	std::cout << "done" << std::endl << "Running kmeans...";
	auto start = std::chrono::high_resolution_clock::now();
	cv::Mat centers, labels;
	cv::kmeans(data, k, labels, cv::TermCriteria(cv::TermCriteria::EPS, 0, 0.01), 1, cv::KMEANS_PP_CENTERS, centers);
	auto end = std::chrono::high_resolution_clock::now();
	std::cout << "done" << std::endl;
	return end - start;
}

int main() {
	std::cout << "# BEGINNING PROFILING #\n" << std::endl;
	auto cv_data = load_opencv();
	auto time_opencv = profile_opencv(cv_data, 3);
	std::cout << "OpenCV: " << std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(time_opencv).count() << "ms" << std::endl;
	
	return 0;
}