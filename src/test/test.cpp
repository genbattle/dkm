// clang-format disabled because clang-format doesn't format lest's macros correctly
// clang-format off
/*
Test cases for dkm.hpp

This is just simple test harness without any external dependencies.
*/

#include "../../include/dkm.hpp"
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


const lest::test specification[] = {
	CASE("Small 2D dataset is successfully segmented into 3 clusters",) {
		SETUP("Small 2D dataset") {
			std::vector<std::array<float, 2>> data{{1.f, 1.f}, {2.f, 2.f}, {1200.f, 1200.f}, {2.f, 2.f}};
			uint32_t k = 3;
			
			SECTION("Distance squared calculated correctly") {
				EXPECT(dkm::details::distance_squared(data[0], data[1]) == lest::approx(2.f));
				EXPECT(dkm::details::distance_squared(data[1], data[2]) == lest::approx(2870408.f));
			}
			
			SECTION("Initial means picked correctly") {
				auto means = dkm::details::random_plusplus(data, k);
				EXPECT(means.size() == 3u);
				// means are 4 values from the input data vector
				uint32_t count = 0;
				for (auto& m : means) {
					for (auto& d : data) {
						if (m == d) {
							++count;
							break;
						}
					}
				}
				EXPECT(count == 3u);
				// means aren't all the same value
				EXPECT((means[0] != means[1] || means[1] != means[2] || means[0] != means[2]));
			}
			
			SECTION("K-means calculated correctly via Lloyds method") {
				auto means_clusters = dkm::kmeans_lloyd(data, 3);
				auto means = std::get<0>(means_clusters);
				auto clusters = std::get<1>(means_clusters);
				// verify results
				EXPECT(means.size() == 3u);
				EXPECT(clusters.size() == data.size());
				std::vector<std::array<float, 2>> expected_means{{1.f, 1.f}, {2.f, 2.f}, {1200.f, 1200.f}};
				std::sort(means.begin(), means.end());
				EXPECT(means == expected_means);
				// Can't verify clusters easily because order may differ from run to run
				// Sorting the means before assigning clusters would help, but would also slow the algorithm down
				EXPECT(std::count(clusters.cbegin(), clusters.cend(), 0) > 0);
				EXPECT(std::count(clusters.cbegin(), clusters.cend(), 1) > 0);
				EXPECT(std::count(clusters.cbegin(), clusters.cend(), 2) > 0);
				EXPECT(std::count(clusters.cbegin(), clusters.cend(), 3) == 0);
			}
		}
	},

	CASE("Test dkm::get_cluster",) {
		SETUP() {
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
				std::vector<std::array<double, 2>> points;
				std::vector<uint32_t> labels;

				SECTION("Empty set of points") {
					auto cluster = dkm::get_cluster(points, labels, 0);
					std::vector<std::array<double, 2>> empty;
					EXPECT(cluster == empty);
				}
			}

			SECTION("points and labels sequences with different sizes") {
				std::vector<std::array<double, 2>> points{
					{0, 1},
					{2, 3.5}
				};
				std::vector<uint32_t> labels{2, 4, 1, 1};
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
				std::vector<std::array<double, 2>> points;
				std::array<double, 2> center{5, 4};

				std::vector<double> empty;
				std::vector<double> out = dkm::dist_to_center(points, center);

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
		}
	}
};

int main(int argc, char** argv) {
	return lest::run(specification, argc, argv);
}
