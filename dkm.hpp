#pragma once

// only included in case there's a C++11 compiler out there that doesn't support `#pragma once`
#ifndef DKM_KMEANS_H
#define DKM_KMEANS_H

#include <vector>
#include <array>
#include <tuple>
#include <cstddef>
#include <cstdint>
#include <random>
#include <type_traits>
#include <cassert>

/*
DKM - A k-means implementation that is generic across variable data dimensions.
*/
namespace dkm {

/*
These functions are all private implementation details and shouldn't be referenced outside of this 
file.
*/
namespace details {

/*
Randomly select initial means from data set (Forgy method).
*/
template <typename T, size_t N>
std::vector<std::array<T, N>> random_sample(const std::vector<std::array<T, N>>& data, uint32_t k) {
	using input_size_t = typename std::array<T,N>::size_type;
	std::vector<std::array<T, N>> means;
	// Using a very simple PRBS generator, parameters selected according to 
	// https://en.wikipedia.org/wiki/Linear_congruential_generator#Parameters_in_common_use
	std::random_device rand_device;
	std::linear_congruential_engine<uint64_t, 6364136223846793005, 1442695040888963407, UINT64_MAX> rand_engine(rand_device());
	std::uniform_int_distribution<input_size_t> uniform_generator(0, data.size());
	for (uint32_t i = 0; i < k; ++i) {
		means.push_back(data[uniform_generator(rand_engine)]);
	}
	return means;
}

/*
Calculate the index of the mean a particular data point is closest to (euclidean distance)
*/
template <typename T, size_t N>
uint32_t closest_mean(const std::array<T, N>& point, const std::vector<std::array<T, N>>& means) {
	assert(!means.empty());
	auto calculate_distance = [](const std::array<T, N>& point, const std::array<T, N>& mean){
		T distance = T();
		for (size_t i = 0; i < point.size(); ++i) {
			T difference = point[i] - mean[i];
			distance += difference * difference;
		}
		return distance;
	};
	T smallest_distance = calculate_distance(point, means[0]);
	typename std::array<T, N>::size_type index = 0;
	T distance;
	for (size_t i = 1; i < means.size(); ++i) {
		distance = calculate_distance(point, means[i]);
		if (distance < smallest_distance) {
			smallest_distance = distance;
			index = i;
		}
	}
	return index;
}

/*
Calculate the index of the mean each data point is closest to (euclidean distance).
TODO: formatting
*/
template <typename T, size_t N>
std::vector<uint32_t> calculate_clusters(const std::vector<std::array<T, N>>& data, const std::vector<std::array<T, N>>& means) {
	std::vector<uint32_t> clusters;
	for (auto& point : data) {
		clusters.push_back(closest_mean(point, means));
	}
	return clusters;
}

/*
Calculate means based on data points and their cluster assignments.
*/
template <typename T, size_t N>
std::vector<std::array<T, N>> calculate_means(const std::vector<std::array<T, N>>& data, const std::vector<uint32_t> clusters, uint32_t k) {
	std::vector<std::array<T, N>> means(k);
	std::vector<T> count(k, T());
	for (size_t i = 0; i < std::min(clusters.size(), data.size()); ++i) {
		auto& mean = means[clusters[i]];
		count[i] += 1;
		for (size_t j = 0; j < std::min(data[i].size(), mean.size()); ++i) {
			mean[j] += data[i][j];
		}
	}
	return means;
}

} // namespace details

/*
Implementation of k-means generic across the data type and the dimension of each data item. Expects
the data to be a vector of fixed-size arrays. Generic parameters are the type of the base data (T)
and the dimensionality of each data point (N). 

e.g. points of the form (X, Y, Z) would be N = 3.

Returns a std::tuple containing:
  0: A vector holding the means for each cluster from 0 to k-1.
  1: A vector containing the cluster number (0 to k-1) for each corresponding element of the input 
     data vector.
  
Implementation details:
This implementation of k-means uses [Lloyd's Algorithm](https://en.wikipedia.org/wiki/Lloyd%27s_algorithm)
with the [Forgy method](https://en.wikipedia.org/wiki/K-means_clustering#Initialization_methods) 
used for initializing the means.
*/
template <typename T, size_t N>
std::tuple<std::vector<std::array<T, N>>, std::vector<uint32_t>> kmeans_lloyd(const std::vector<std::array<T, N>>& data, uint32_t k) {
	static_assert(std::is_arithmetic<T>::value && std::is_signed<T>::value, "kmeans_lloyd requires the template parameter T to be a signed arithmetic type (e.g. float, double, int)");
	assert(k > 0); // k must be greater than zero
	assert(data.size() >= k); // there must be at least k data points
	std::vector<std::array<T, N>> means = details::random_sample(data, k);
	
	std::vector<std::array<T, N>> new_means;
	std::vector<uint32_t> clusters;
	// Calculate new means until convergence is reached
	while (means != new_means) {
		clusters = details::calculate_clusters(data, means);
		new_means = details::calculate_means(data, clusters, k);
	}
	
	return std::tuple<std::vector<std::array<T, N>>, std::vector<uint32_t>>(means, clusters);
}

} // namespace dkm

#endif /* DKM_KMEANS_H */
