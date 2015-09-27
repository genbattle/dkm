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

/*
DKM - A k-means implementation that is generic across variable data dimensions.
*/
namespace dkm {
/*
Implementation of k-means generic across the data type and the dimension of each data item. Expects
the data to be a vector of fixed-size arrays. Generic parameters are the type of the base data (T)
and the dimensionality of each data point (N). 

e.g. points of the form (X, Y, Z) would be N = 3.

Returns a std::tuple containing:
  0: A vector holding the means for each cluster from 0 to k-1.
  1: A vector containing the cluster number (0 to k-1) for each corresponding element of the input data vector.
  
Implementation details:
This implementation of k-means uses [Lloyd's Algorithm](https://en.wikipedia.org/wiki/Lloyd%27s_algorithm)
with the [Forgy method](https://en.wikipedia.org/wiki/K-means_clustering#Initialization_methods) 
used for initializing the means.
*/
template <typename T, size_t N>
std::tuple<std::vector<std::array<T, N>>, std::vector<uint32_t>> kmeans_lloyd(const std::vector<std::array<T, N>>& data, uint32_t k) {
	using input_size_t = typename std::array<T,N>::size_type;
	std::vector<std::array<T, N>> means;
	// Randomly select initial means from data set (Forgy method)
	{
		// Using a very simple PRBS generator, parameters selected according to 
		// https://en.wikipedia.org/wiki/Linear_congruential_generator#Parameters_in_common_use
		std::random_device rand_device;
		std::linear_congruential_engine<uint64_t, 6364136223846793005, 1442695040888963407, UINT64_MAX> rand_engine(rand_device());
		std::uniform_int_distribution<input_size_t> uniform_generator(0, data.size());
		for (uint32_t i = 0; i < k; ++i) {
			means.push_back(data[uniform_generator(rand_engine)]);
		}
	}
	
	// TODO: Implementation
	return std::tuple<std::vector<std::array<T, N>>, std::vector<uint32_t>>(means, std::vector<uint32_t>());
}

}

#endif /* DKM_KMEANS_H */
