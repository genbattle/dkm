// clang-format disabled because clang-format doesn't format lest's macros correctly
// clang-format off
/*
Test cases for dkm.hpp

This is just simple test harness without any external dependencies.
*/

#include "../../include/dkm.hpp"
#include "../../include/dkm_parallel.hpp"
#include "../../include/dkm_utils.hpp"
#include "lest.hpp"

#include <vector>
#include <array>
#include <cstdint>
#include <algorithm>
#include <tuple>

#ifdef __clang__
#pragma clang diagnostic ignored "-Wmissing-braces"
#endif

constexpr uint64_t random_seed_value = 7;

const lest::test specification[] = {
	CASE("Small 2D dataset is successfully segmented into 3 clusters",) {
		SETUP("Small 2D dataset") {
			std::vector<std::array<float, 2>> data{
				{18.789f, 19.684f},
				{-41.478f, -19.799f},
				{-22.410f, -6.794f},
				{-29.411f  , -8.416f},
				{194.874f, 6.187f},
				{86.881f, 34.023f},
				{125.640f, 24.364f},
				{14.900f, 29.114f},
				{15.082f, 23.051f},
				{-24.638f, -7.013f},
				{-26.608f, -23.007f},
				{-31.118f, -11.876f},
				{-24.734f, -3.788f},
				{133.423f, 23.644f},
				{14.346f, 21.789f},
				{16.875f, 23.290f},
				{132.308f, -0.032f}
			};

			// means: [17,27], [-27, -12], [128, 10]
			dkm::clustering_parameters<float> parameters(3);
			parameters.set_random_seed(random_seed_value);
			
			SECTION("Distance squared calculated correctly") {
				EXPECT(dkm::details::distance_squared(data[0], data[1]) == lest::approx(5191.02f));
				EXPECT(dkm::details::distance_squared(data[1], data[2]) == lest::approx(532.719f));
			}
			
			SECTION("Initial means picked correctly") {
				auto means = dkm::details::random_plusplus(data, parameters.get_k(), parameters.get_random_seed());
				std::vector<std::array<float, 2>> expected_means{{15.082f, 23.051f}, {133.423f, 23.644f}, {-24.734f, -3.788f}};
				EXPECT(means.size() == 3u);
				EXPECT(means == expected_means);
			}
			
			SECTION("K-means calculated correctly via Lloyds method") {
				auto means_clusters = dkm::kmeans_lloyd(data, parameters);
				auto means = std::get<0>(means_clusters);
				auto clusters = std::get<1>(means_clusters);
				// verify results
				std::vector<std::array<float, 2>> expected_means{{15.9984f, 23.3856f}, {134.625f, 17.6372f}, {-28.6281f, -11.5276f}};
				EXPECT(means.size() == 3u);
				for (size_t i = 0; i < means.size(); ++i) {
					for (size_t j = 0; j < means[i].size(); ++j) {
						EXPECT(means[i][j] == lest::approx(expected_means[i][j]));
					}
				}
				std::vector<uint32_t> expected_clusters = { 0, 2, 2, 2, 1, 1, 1, 0, 0, 2, 2, 2, 2, 1, 0, 0, 1};
				EXPECT(clusters.size() == data.size());
				EXPECT(clusters == expected_clusters);
			}

			SECTION("K-means calculated correctly via parallel Lloyds method") {
				auto means_clusters = dkm::kmeans_lloyd_parallel(data, parameters);
				auto means = std::get<0>(means_clusters);
				auto clusters = std::get<1>(means_clusters);
				// verify results
				EXPECT(means.size() == 3u);
				EXPECT(clusters.size() == data.size());
				std::vector<std::array<float, 2>> expected_means{{15.9984f, 23.3856f}, {134.625f, 17.6372f}, {-28.6281f, -11.5276f}};
				EXPECT(means.size() == 3u);
				for (size_t i = 0; i < means.size(); ++i) {
					for (size_t j = 0; j < means[i].size(); ++j) {
						EXPECT(means[i][j] == lest::approx(expected_means[i][j]));
					}
				}
				std::vector<uint32_t> expected_clusters = { 0, 2, 2, 2, 1, 1, 1, 0, 0, 2, 2, 2, 2, 1, 0, 0, 1};
				EXPECT(clusters.size() == data.size());
				EXPECT(clusters == expected_clusters);
			}
		}
	},

	CASE("Test with real data set",) {
		SETUP("Real data set") {
			auto data = dkm::load_csv<float, 2>("iris.data.csv");
			dkm::clustering_parameters<float> parameters(3);
			parameters.set_random_seed(random_seed_value);

			SECTION("Segmentation completes to convergence") {
				auto means_clusters = dkm::kmeans_lloyd(data, parameters);
				auto means = std::get<0>(means_clusters);
				auto clusters = std::get<1>(means_clusters);

				EXPECT(means.size() == 3u);
				EXPECT(clusters.size() == data.size());
				std::vector<std::array<float, 2>> expected_means {
					{3.44082f, 0.242857f},
					{2.70755f, 1.30943f},
					{3.04167f, 2.05208f},
				};
				for (size_t i = 0; i < means.size(); ++i) {
					for (size_t j = 0; j < means[i].size(); ++j) {
						EXPECT(means[i][j] == lest::approx(expected_means[i][j]));
					}
				}
				// not checking clusters here because there are too many points
			}

			SECTION("Segmentation completes to convergence with parallel implementation") {
				auto means_clusters = dkm::kmeans_lloyd_parallel(data, parameters);
				auto means = std::get<0>(means_clusters);
				auto clusters = std::get<1>(means_clusters);

				EXPECT(means.size() == 3u);
				EXPECT(clusters.size() == data.size());
				std::vector<std::array<float, 2>> expected_means {
					{3.44082f, 0.242857f},
					{2.70755f, 1.30943f},
					{3.04167f, 2.05208f},
				};
				for (size_t i = 0; i < means.size(); ++i) {
					for (size_t j = 0; j < means[i].size(); ++j) {
						EXPECT(means[i][j] == lest::approx(expected_means[i][j]));
					}
				}
				// not checking clusters here because there are too many points
			}

			SECTION("Segmentation completes early because iteration limit is reached") {
				parameters.set_max_iteration(5);
				auto means_clusters = dkm::kmeans_lloyd(data, parameters);
				auto means = std::get<0>(means_clusters);
				auto clusters = std::get<1>(means_clusters);

				EXPECT(means.size() == 3u);
				EXPECT(clusters.size() == data.size());
				std::vector<std::array<float, 2>> expected_means {
					{3.418f, 0.244f},
					{2.72857f, 1.41587f},
					{3.11622f, 2.11892f},
				};
				for (size_t i = 0; i < means.size(); ++i) {
					for (size_t j = 0; j < means[i].size(); ++j) {
						EXPECT(means[i][j] == lest::approx(expected_means[i][j]));
					}
				}
				// not checking clusters here because there are too many points
			}

			SECTION("Segmentation completes early because iteration limit is reached with parallel implementation") {
				parameters.set_max_iteration(5);
				auto means_clusters = dkm::kmeans_lloyd_parallel(data, parameters);
				auto means = std::get<0>(means_clusters);
				auto clusters = std::get<1>(means_clusters);

				EXPECT(means.size() == 3u);
				EXPECT(clusters.size() == data.size());
				std::vector<std::array<float, 2>> expected_means {
					{3.418f, 0.244f},
					{2.72857f, 1.41587f},
					{3.11622f, 2.11892f},
				};
				for (size_t i = 0; i < means.size(); ++i) {
					for (size_t j = 0; j < means[i].size(); ++j) {
						EXPECT(means[i][j] == lest::approx(expected_means[i][j]));
					}
				}
				// not checking clusters here because there are too many points
			}
		}
	},

	CASE("Test with uniform data points",) {
		SETUP("Uniform data points") {
			std::vector<std::array<float, 2>> data{
				{5, 5},
				{5, 5},
				{5, 5},
				{5, 5},
				{5, 5},
				{5, 5},
				{5, 5},
				{5, 5},
				{5, 5},
				{5, 5},
			};
			dkm::clustering_parameters<float> parameters(1);
			parameters.set_random_seed(random_seed_value);

			SECTION("KMeans++ doesn't throw an exception on uniform data") {
				auto means_clusters = dkm::details::random_plusplus(data, parameters.get_k(), parameters.get_random_seed());

				std::vector<std::array<float, 2>> expected_means{{5.0f, 5.0f}};
				EXPECT(means_clusters.size() == 1u);
				EXPECT(means_clusters == expected_means);
			}
		}
	},

	CASE("Test with unsigned integer data type",) {
		SETUP("Unsigned integer data") {
			std::vector<std::array<uint32_t, 2>> data{
				{0, 0},
				{1, 1},
				{2, 2},
				{3, 3},
				{4, 4},
				{5, 5},
				{6, 6},
				{7, 7},
				{8, 8},
				{9, 9},
			};
			dkm::clustering_parameters<uint32_t> parameters(3);
			parameters.set_random_seed(random_seed_value);

			SECTION("Test clustering works with unsigned values") {
				auto means_clusters = dkm::kmeans_lloyd(data, parameters);
				auto means = std::get<0>(means_clusters);
				auto clusters = std::get<1>(means_clusters);

				EXPECT(means.size() == 3u);
				EXPECT(clusters.size() == data.size());
				std::vector<std::array<float, 2>> expected_means {
					{1, 1},
					{8, 8},
					{5, 5},
				};
				std::vector<uint32_t> expected_clusters{0, 0, 0, 0, 2, 2, 2, 1, 1, 1};
				EXPECT(clusters == expected_clusters);
				for (size_t i = 0; i < means.size(); ++i) {
					for (size_t j = 0; j < means[i].size(); ++j) {
						EXPECT(means[i][j] == lest::approx(expected_means[i][j]));
					}
				}
			}
		}
	},

	CASE("Test dkm::get_cluster",) {
		SETUP("Linear data for get_cluster test") {
			std::vector<std::array<double, 2>> points{
				{0, 0},
				{1, 1},
				{2, 2},
				{3, 3},
				{4, 4},
				{5, 5},
				{6, 6},
				{7, 7},
				{8, 8},
				{9, 9},
			};
			std::vector<uint32_t> labels{0, 2, 1, 1, 0, 2, 2, 1, 1, 0};
			SECTION("Non-empty and same size points and labels") {

				SECTION("Correct points for existing labels") {
					auto cluster = dkm::get_cluster(points, labels, 0);
					std::vector<std::array<double, 2>> res{
						{0, 0},
							{4, 4},
							{9, 9}
					};
					EXPECT(cluster == res);

					cluster = dkm::get_cluster(points, labels, 1);
					res = {
						{2, 2},
						{3, 3},
						{7, 7},
						{8, 8}
					};
					EXPECT(cluster == res);

					cluster = dkm::get_cluster(points, labels, 2);
					res = {
						{1, 1},
						{5, 5},
						{6, 6},
					};
					EXPECT(cluster == res);
				}

				SECTION("Empty set of points for non-existing labels") {
					auto cluster = dkm::get_cluster(points, labels, 4);
					std::vector<std::array<double, 2>> empty;
					EXPECT(cluster == empty);
				}
			}

			SECTION("Empty points and labels") {
				std::vector<std::array<double, 2>> points_blank;
				std::vector<uint32_t> labels;

				SECTION("Empty set of points") {
					auto cluster = dkm::get_cluster(points_blank, labels, 0);
					std::vector<std::array<double, 2>> empty;
					EXPECT(cluster == empty);
				}
			}
		}
	},

	CASE("Test dkm::dist_to_center",) {
		SETUP() {
			std::vector<std::array<double, 2>> points{
				{1, 5},
				{2.2, 3},
				{8, 12},
				{11.4, 4.87},
				{0.27, 50},
				{1, 1}
			};
			std::array<double, 2> center{17.2, 24.5};

			std::vector<double> res{25.3513, 26.2154, 15.5206, 20.4689, 30.6084, 28.5427};
			SECTION("Non-empty sequence of points") {

				std::vector<double> out = dkm::dist_to_center(points, center);

				for (size_t i = 0; i < out.size(); ++i)
					EXPECT(lest::approx(out[i]) == res[i]);
			}

			SECTION("Empty sequence of points returns an empty vector") {
				std::vector<std::array<double, 2>> points_empty;
				std::array<double, 2> center_empty{5, 4};

				std::vector<double> empty;
				std::vector<double> out = dkm::dist_to_center(points_empty, center_empty);

				EXPECT(out == empty);
			}
		}
	},

	CASE("Test dkm::sum_dist",) {
		SETUP() {
			std::vector<std::array<double, 2>> points{
				{1,    5},
				{2.2,  3},
				{8,    12},
				{11.4, 4.87},
				{0.27, 50},
				{1,    1}
			};
			std::vector<double> out(points.size());
			std::array<double, 2> center{17.2, 24.5};

			std::vector<double> res{25.3513, 26.2154, 15.5206, 20.4689, 30.6084, 28.5427};
			SECTION("Non-empty sequence of points") {

				EXPECT(dkm::sum_dist(points, center) == lest::approx(146.7073));
			}

			SECTION("Empty sequence of points returns 0") {
				std::vector<std::array<double, 2>> points;
				std::array<double, 2> center{5, 4};

				EXPECT(dkm::sum_dist(points, center) == 0);
			}
		}
	},

	CASE("Test dkm::means_inertia",) {
		SETUP() {
			std::vector<std::array<double, 2>> points{
				{66.01742226,  48.70477854},
				{62.30094932, 108.44049522},
				{39.60740312,  12.07668535},
				{35.57096194,  -7.10722525},
				{39.90890238,  61.89509695},
				{27.5850295 ,  85.50226002},
				{51.14012591,  27.90650051},
				{58.6414776 ,  31.97020798},
				{14.75127435,  69.36707669},
				{73.66255253,  84.73455103},
				{-1.31034384,  66.10406579},
				{41.91865987,  56.5003107 },
				{33.31116528,  45.92203855},
				{57.12362692,  37.73753163},
				{ 2.68915431,  51.35514789},
				{39.76543196,  -5.99499795},
				{72.64312341,  61.43756623},
				{30.97140948,  29.49960625},
				{25.31232669,  35.88059477},
				{57.67046396,  35.05019015}
			};
			std::vector<std::array<double, 2>> centroids{
				{10, 10},
				{20, 20},
				{40, 30}
			};
			std::vector<uint32_t> labels{
				0, 0, 1, 2, 2, 1, 1, 0, 0, 0,
				1, 1, 2, 1, 0, 0, 1, 2, 1, 0
			};
			uint32_t k = 3;
			SECTION("Non-empty set of points, fixed 3 clusters") {
				std::tuple<std::vector<std::array<double, 2>>, std::vector<uint32_t>> means{centroids, labels};

				double inertia = 0;
				for (size_t i = 0; i < labels.size(); ++i) {
					auto center = centroids[labels[i]];
					auto point = points[i];
					inertia += dkm::details::distance(point, center);
				}

				EXPECT(lest::approx(inertia) == dkm::means_inertia(points, means, k));
			}

			SECTION("Empty set of points should give 0 inertia") {
				std::vector<std::array<double, 2>> points;
				std::tuple<std::vector<std::array<double, 2>>, std::vector<uint32_t>> means;

				EXPECT(dkm::means_inertia(points, means, k) == lest::approx(0));
			}

			SECTION() {
				std::vector<std::array<double, 2>> data{
					{1, 1},
						{2, 2},
						{1200, 1200},
						{1000, 1000}
				};
				uint32_t k = 2;
				auto means = dkm::kmeans_lloyd(data, k);
				double inertia = dkm::means_inertia(data, means, k);
				EXPECT(284.256926 == lest::approx(inertia).epsilon(1e-6));
			}
		}
	},
	CASE("Test dkm::get_best_means",) {
		SETUP() {
			std::vector<std::array<double, 2>> points{
				{8,  8},
				{9, 9},
				{11,  11},
				{12,  12},
				{18,  18},
				{19,  19},
				{21,  21},
				{22,  22},
				{39,  39},
				{41,  41},
			};
			std::vector<std::array<double, 2>> centroids{
				{10, 10},
				{20, 20},
				{40, 40}
			};
			std::vector<uint32_t> labels{0, 0, 0, 0, 1, 1, 1, 1, 2, 2};
			uint32_t k = 3;
			SECTION("Test if we get the clustering with the least inertia") {
				auto means = dkm::get_best_means(points, k, 20);
				std::vector<std::array<double, 2>> returned_centroids;
				std::vector<uint32_t> returned_labels;
				std::tie(returned_centroids, returned_labels) = means;
				// every point is assigned to the same cluster center
				for (uint32_t i = 0; i < points.size(); ++i) {
					auto expected_center = centroids[labels[i]];
					auto returned_center = returned_centroids[returned_labels[i]];
					EXPECT(expected_center[0] == lest::approx(returned_center[0]));
					EXPECT(expected_center[1] == lest::approx(returned_center[1]));
				}
			}
		}
	},
	CASE("Test dkm::predict",) {
		SETUP() {
			std::vector<std::array<double, 2>> centroids{
					{8,  8},
					{9, 9},
					{11,  11},
					{12,  12},
					{18,  18},
					{19,  19},
					{21,  21},
					{22,  22},
					{39,  39},
					{41,  41},
			};
			std::array<double, 2> query {11, 10.5};
			SECTION("Test if we get the actual closest centroid to the query") {
				auto res = dkm::predict(centroids, query);
				EXPECT(res == 2u);
			}
		}
	}
};

int main(int argc, char** argv) {
	return lest::run(specification, argc, argv);
}
