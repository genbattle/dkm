#pragma once

#include <algorithm>
#include <array>
#include <tuple>
#include <vector>

#include "dkm.hpp"

namespace dkm {


/**
 * Calculates the Euclidean distance from each point in the given sequence
 * to given center and returns the results as a vector.
 *
 * @param points Point sequence.
 * @param center Center point with which the distance of each point is calculated.
 *
 * @return std::vector<T> containing distance of each point to the center.
 */
template <typename T, size_t N>
std::vector<T> dist_to_center(const std::vector<std::array<T, N>>& points, const std::array<T, N>& center) {
	std::vector<T> result(points.size());
	std::transform(points.begin(), points.end(), result.begin(), [&center](const std::array<T, N>& p) {
		return details::distance(p, center);
	});
	return result;
}


/**
 * Calculates sum of distances from each point in points to given center point.
 *
 * @param points Point sequence.
 * @param center Center point with which the distance of each point is calculated.
 *
 * @return Sum of distances of each point to the center.
 */
template <typename T, size_t N>
T sum_dist(const std::vector<std::array<T, N>>& points, const std::array<T, N>& center) {
	std::vector<T> distances = dist_to_center(points, center);
	return std::accumulate(distances.begin(), distances.end(), T());
}


/**
 * Return a point sequence whose elements all belong to the same cluster given
 * by label.
 *
 * @param points Sequence that were passed to dkm::kmeans_lloyd
 * @param labels Sequence of labels that were obtained from dkm:kmeans_lloyd
 * @param label  Label of the cluster to be obtained.
 *
 * @return Sequence of points that all belong to the cluster with the given label.
 */
template <typename T, size_t N>
std::vector<std::array<T, N>> get_cluster(
	const std::vector<std::array<T, N>>& points, const std::vector<uint32_t>& labels, const uint32_t label) {
	assert(points.size() == labels.size() && "Points and labels have different sizes");
	// construct the cluster
	std::vector<std::array<T, N>> cluster;
	for (size_t point_index = 0; point_index < points.size(); ++point_index) {
		if (labels[point_index] == label) {
			cluster.push_back(points[point_index]);
		}
	}
	return cluster;
}


/**
 * Calculates inertia of a given k-means cluster. Inertia is defined as sum of
 * Euclidean distances of each point to its closest cluster center.
 *
 * @param points Sequence that were passed to dkm::kmeans_lloyd
 * @param means  Result of dkm::kmeans_lloyd
 * @param k      Number of clusters
 *
 * @return Total inertia of the given clustering.
 */
template <typename T, size_t N>
T means_inertia(const std::vector<std::array<T, N>>& points,
	const std::tuple<std::vector<std::array<T, N>>, std::vector<uint32_t>>& means,
	uint32_t k) {
	std::vector<std::array<T, N>> centroids;
	std::vector<uint32_t> labels;
	std::tie(centroids, labels) = means;

	T inertia{T()};
	for (uint32_t i = 0; i < k; ++i) {
		auto cluster = get_cluster(points, labels, i);
		inertia += sum_dist(cluster, centroids[i]);
	}
	return inertia;
}


/**
 * Return the best clustering obtained from a given number of k-means
 * calculations.
 *
 * @param points  Sequence of points to be clustered.
 * @param k		  Number of clusters
 * @param n_init  Number of times a k-means clustering will be calculated.
 *
 * @return Clustering with the lowest inertia.
 */
template <typename T, size_t N>
std::tuple<std::vector<std::array<T, N>>, std::vector<uint32_t>> get_best_means(
	const std::vector<std::array<T, N>>& points, uint32_t k, uint32_t n_init = 10) {
	auto best_means = kmeans_lloyd(points, k);
	auto best_inertia = means_inertia(points, best_means, k);

	for (uint32_t i = 0; i < n_init - 1; ++i) {
		auto curr_means = kmeans_lloyd(points, k);
		auto curr_inertia = means_inertia(points, curr_means, k);
		if (curr_inertia < best_inertia) {
			best_inertia = curr_inertia;
			best_means = curr_means;
		}
	}
	// copy and return
	return best_means;
}

} // namespace dkm
