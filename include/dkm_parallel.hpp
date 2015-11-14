#pragma once

// only included in case there's a C++11 compiler out there that doesn't support `#pragma once`
#ifndef DKM_KMEANS_PARALLEL_H
#define DKM_KMEANS_PARALLEL_H

#include "dkm.hpp"

#include <vector>
#include <array>
#include <tuple>
#include <cstddef>
#include <cstdint>
#include <random>
#include <type_traits>
#include <cassert>
#include <future>

/*
DKM - A k-means implementation that is generic across variable data dimensions.
*/
namespace dkm_parallel {

/*
These functions are all private implementation details and shouldn't be referenced outside of this 
file.
*/
namespace details {

/*
Calculate means based on data points and their cluster assignments.
*/
template <typename T, size_t N>
std::vector<std::array<T, N>> calculate_means(const std::vector<std::array<T, N>>& data, const std::vector<uint32_t>& clusters, const std::vector<std::array<T, N>>& old_means, uint32_t k) {
	// TODO: filter data for each cluster
	std::vector<std::future<std::array<T, N>>> mf;
	mf.reserve(old_means.size());
	auto calc_mean = [](uint32_t cluster, const std::array<T, N>& old_mean, const std::vector<std::array<T, N>>& data, const std::vector<uint32_t>& clusters) -> std::array<T, N> {
		auto mean = old_mean;
		size_t count = 0;
		for (size_t j = 0; j < std::min(clusters.size(), data.size()); ++j) {
			if (clusters[j] == cluster) {
				for (size_t h = 0; h < std::min(data[j].size(), mean.size()); ++h) {
					mean[h] = ((mean[h] * count) + data[j][h]) / T(count + 1);
				}
				++count;
			}
		}
		return mean;
	};
	for (uint32_t i = 0; i < k; ++i) {
		assert(k == old_means.size());
		mf.push_back(std::async(std::launch::async, calc_mean, i, old_means[i], data, clusters));
	}
	
	std::vector<std::array<T, N>> means;
	means.reserve(old_means.size());
	for (auto& f : mf) {
		means.push_back(f.get());
	}
	return means;
}

} // namespace details

/*
Implementation of k-means generic across the data type and the dimension of each data item. Expects
the data to be a vector of fixed-size arrays. Generic parameters are the type of the base data (T)
and the dimensionality of each data point (N). All points must have the same dimensionality. 

e.g. points of the form (X, Y, Z) would be N = 3.

Returns a std::tuple containing:
  0: A vector holding the means for each cluster from 0 to k-1.
  1: A vector containing the cluster number (0 to k-1) for each corresponding element of the input 
     data vector.
  
Implementation details:
This implementation of k-means uses [Lloyd's Algorithm](https://en.wikipedia.org/wiki/Lloyd%27s_algorithm)
with the [kmeans++](https://en.wikipedia.org/wiki/K-means%2B%2B) 
used for initializing the means.

TODO: formatting
*/
template <typename T, size_t N>
std::tuple<std::vector<std::array<T, N>>, std::vector<uint32_t>> kmeans_lloyd(const std::vector<std::array<T, N>>& data, uint32_t k) {
	static_assert(std::is_arithmetic<T>::value && std::is_signed<T>::value, "kmeans_lloyd requires the template parameter T to be a signed arithmetic type (e.g. float, double, int)");
	assert(k > 0); // k must be greater than zero
	assert(data.size() >= k); // there must be at least k data points
	std::vector<std::array<T, N>> means = dkm::details::random_plusplus(data, k);
	
	std::vector<std::array<T, N>> old_means;
	std::vector<uint32_t> clusters;
	// Calculate new means until convergence is reached
	int count = 0;
	do {
		clusters = dkm::details::calculate_clusters(data, means);
		old_means = means;
		means = details::calculate_means(data, clusters, old_means, k);
		++count;
	} while (means != old_means);
	
	return std::tuple<std::vector<std::array<T, N>>, std::vector<uint32_t>>(means, clusters);
}

} // namespace dkm_parallel

#endif /* DKM_KMEANS_PARALLEL_H */
