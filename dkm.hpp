/*
DKM - A k-means implementation that is generic across variable data dimensions.
*/

#include <vector>
#include <array>
#include <cstddef>
#include <cstdint>

/*
Implementation of k-means generic across the data type and the dimension of each data item. Expects
the data to be a vector of fixed-size arrays. Generic parameters are the type of the base data (T)
and the dimensionality of each data point (N). 

e.g. points of the form (X, Y, Z) would be N = 3.

Returns a std::tuple containing:
  0: A vector holding the means for each cluster from 0 to k.
  1: A vector containing the cluster number for each corresponding element of the input data vector.
*/
template <typename T, size_t N>
std::tuple<std::vector<T>, std::vector<uint32_t>> kmeans(const std::vector<std::array<T,N>>& data, uint32_t k) {
	// TODO: Implementation
	return std::tuple<std::vector<T>, std::vector<uint32_t>>();
}

