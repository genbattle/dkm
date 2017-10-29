#pragma once

#include <algorithm>
#include <array>
#include <tuple>
#include <vector>

#include "dkm.hpp"

namespace dkm {

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
std::vector<std::array<T, N>> get_cluster(const std::vector<std::array<T, N>>& points,
										  const std::vector<uint32_t>& labels,
										  const uint32_t label)
{
	if (points.size() != labels.size())
		throw std::runtime_error("points and labels have different sizes");

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
 * @param points Result of dkm::kmeans_lloyd
 *
 * @return Total inertia of the given clustering.
 */
template <typename T, size_t N>
T means_inertia(const std::vector<std::array<T, N>>& points,
					 const std::tuple<std::vector<std::array<T, N>>, std::vector<uint32_t>>& means)
{
	std::vector<std::array<T, N>> centroids;
	std::vector<uint32_t> labels;
	std::tie(centroids, labels) = means;

	// get a list of unique labels
	std::vector<uint32_t> labels_copy(labels.size());
	std::copy(labels.begin(), labels.end(), labels_copy.begin());
	std::sort(labels_copy.begin(), labels_copy.end());
	auto uniq_labels_end = std::unique(labels_copy.begin(), labels_copy.end());

	double inertia = 0;
	for (auto it = labels_copy.begin(); it != uniq_labels_end; ++it) {
		auto label = *it;
		auto cluster = get_cluster(points, labels, label);
		inertia += sum_dist(cluster, centroids[label]);
	}
	return inertia;
}


/**
 * Returns the best dkm_means object from the means_list based on inertia.
 *
 * @param points     Sequence of points that were passed to dkm k-means
 *					 clustering algorithm.
 *
 * @param means_list Sequence of dkm means.
 *
 * @return dkm means with the lowest inertia.
 */
template <typename T, size_t N>
std::tuple<std::vector<std::array<T, N>>, std::vector<uint32_t>>
get_best_means(const std::vector<std::array<T, N>>& points,
			   const std::vector<std::tuple<std::vector<std::array<T, N>>, std::vector<uint32_t>>>& means_list)
{
	double min_inertia = std::numeric_limits<double>::max();
	const std::tuple<std::vector<std::array<T, N>>, std::vector<uint32_t>>* best_means_ptr = nullptr;

	for (const auto& means : means_list) {
		double inertia = means_inertia(points, means);
		if (inertia < min_inertia) {
			min_inertia = inertia;
			best_means_ptr = &means;
		}
	}
	// copy and return
	return std::tuple<std::vector<std::array<T, N>>, std::vector<uint32_t>>{*best_means_ptr};
}


/**
 * Run dkm::kmeans_lloyd on the given sequence of points n times and return the best
 * means with the lowest inertia.
 *
 * @param points    Poinst to be sent to dkm::kmeans_lloyd.
 * @param k			Number of clusters.
 * @param n_init    Number of times dkm::kmeans_lloyd algorithm will be run.
 *
 * @return clustering with the lowest inertia.
 */
template <typename T, size_t N>
std::tuple<std::vector<std::array<T, N>, std::vector<uint32_t>>>
n_kmeans(const std::vector<std::array<T, N>>& points, uint32_t k, uint32_t n_init=10)
{
	// Run k-means algorithm n_init times and collect the results.
	std::vector<std::tuple<std::vector<std::array<T, N>, std::vector<uint32_t>>>> means_list;
	for (uint32_t i = 0; i < n_init; ++i) {
		means_list.push_back(dkm::kmeans_lloyd(points, k));
	}

	// Return the best k-means result.
	return get_best_means(points, means_list);
}

} // dkm namespace
