/*
Test cases for dkm.hpp

This is just simple test harness without any external dependencies.
*/

#include "../include/dkm.hpp"
#include "lest.hpp"

#include <vector>
#include <array>
#include <cstdint>
#include <algorithm>
#include <tuple>



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
	
	CASE("2D iris data set segmented into 3 clusters",) {
		// Iris data taken from https://archive.ics.uci.edu/ml/datasets/Iris
		// Filtered for only petal width and sepal width
		std::vector<std::array<float, 4>> iris_data = {
			{5.1,3.5,1.4,0.2},
			{4.9,3,1.4,0.2},
			{4.7,3.2,1.3,0.2},
			{4.6,3.1,1.5,0.2},
			{5,3.6,1.4,0.2},
			{5.4,3.9,1.7,0.4},
			{4.6,3.4,1.4,0.3},
			{5,3.4,1.5,0.2},
			{4.4,2.9,1.4,0.2},
			{4.9,3.1,1.5,0.1},
			{5.4,3.7,1.5,0.2},
			{4.8,3.4,1.6,0.2},
			{4.8,3,1.4,0.1},
			{4.3,3,1.1,0.1},
			{5.8,4,1.2,0.2},
			{5.7,4.4,1.5,0.4},
			{5.4,3.9,1.3,0.4},
			{5.1,3.5,1.4,0.3},
			{5.7,3.8,1.7,0.3},
			{5.1,3.8,1.5,0.3},
			{5.4,3.4,1.7,0.2},
			{5.1,3.7,1.5,0.4},
			{4.6,3.6,1,0.2},
			{5.1,3.3,1.7,0.5},
			{4.8,3.4,1.9,0.2},
			{5,3,1.6,0.2},
			{5,3.4,1.6,0.4},
			{5.2,3.5,1.5,0.2},
			{5.2,3.4,1.4,0.2},
			{4.7,3.2,1.6,0.2},
			{4.8,3.1,1.6,0.2},
			{5.4,3.4,1.5,0.4},
			{5.2,4.1,1.5,0.1},
			{5.5,4.2,1.4,0.2},
			{4.9,3.1,1.5,0.1},
			{5,3.2,1.2,0.2},
			{5.5,3.5,1.3,0.2},
			{4.9,3.1,1.5,0.1},
			{4.4,3,1.3,0.2},
			{5.1,3.4,1.5,0.2},
			{5,3.5,1.3,0.3},
			{4.5,2.3,1.3,0.3},
			{4.4,3.2,1.3,0.2},
			{5,3.5,1.6,0.6},
			{5.1,3.8,1.9,0.4},
			{4.8,3,1.4,0.3},
			{5.1,3.8,1.6,0.2},
			{4.6,3.2,1.4,0.2},
			{5.3,3.7,1.5,0.2},
			{5,3.3,1.4,0.2},
			{7,3.2,4.7,1.4},
			{6.4,3.2,4.5,1.5},
			{6.9,3.1,4.9,1.5},
			{5.5,2.3,4,1.3},
			{6.5,2.8,4.6,1.5},
			{5.7,2.8,4.5,1.3},
			{6.3,3.3,4.7,1.6},
			{4.9,2.4,3.3,1},
			{6.6,2.9,4.6,1.3},
			{5.2,2.7,3.9,1.4},
			{5,2,3.5,1},
			{5.9,3,4.2,1.5},
			{6,2.2,4,1},
			{6.1,2.9,4.7,1.4},
			{5.6,2.9,3.6,1.3},
			{6.7,3.1,4.4,1.4},
			{5.6,3,4.5,1.5},
			{5.8,2.7,4.1,1},
			{6.2,2.2,4.5,1.5},
			{5.6,2.5,3.9,1.1},
			{5.9,3.2,4.8,1.8},
			{6.1,2.8,4,1.3},
			{6.3,2.5,4.9,1.5},
			{6.1,2.8,4.7,1.2},
			{6.4,2.9,4.3,1.3},
			{6.6,3,4.4,1.4},
			{6.8,2.8,4.8,1.4},
			{6.7,3,5,1.7},
			{6,2.9,4.5,1.5},
			{5.7,2.6,3.5,1},
			{5.5,2.4,3.8,1.1},
			{5.5,2.4,3.7,1},
			{5.8,2.7,3.9,1.2},
			{6,2.7,5.1,1.6},
			{5.4,3,4.5,1.5},
			{6,3.4,4.5,1.6},
			{6.7,3.1,4.7,1.5},
			{6.3,2.3,4.4,1.3},
			{5.6,3,4.1,1.3},
			{5.5,2.5,4,1.3},
			{5.5,2.6,4.4,1.2},
			{6.1,3,4.6,1.4},
			{5.8,2.6,4,1.2},
			{5,2.3,3.3,1},
			{5.6,2.7,4.2,1.3},
			{5.7,3,4.2,1.2},
			{5.7,2.9,4.2,1.3},
			{6.2,2.9,4.3,1.3},
			{5.1,2.5,3,1.1},
			{5.7,2.8,4.1,1.3},
			{6.3,3.3,6,2.5},
			{5.8,2.7,5.1,1.9},
			{7.1,3,5.9,2.1},
			{6.3,2.9,5.6,1.8},
			{6.5,3,5.8,2.2},
			{7.6,3,6.6,2.1},
			{4.9,2.5,4.5,1.7},
			{7.3,2.9,6.3,1.8},
			{6.7,2.5,5.8,1.8},
			{7.2,3.6,6.1,2.5},
			{6.5,3.2,5.1,2},
			{6.4,2.7,5.3,1.9},
			{6.8,3,5.5,2.1},
			{5.7,2.5,5,2},
			{5.8,2.8,5.1,2.4},
			{6.4,3.2,5.3,2.3},
			{6.5,3,5.5,1.8},
			{7.7,3.8,6.7,2.2},
			{7.7,2.6,6.9,2.3},
			{6,2.2,5,1.5},
			{6.9,3.2,5.7,2.3},
			{5.6,2.8,4.9,2},
			{7.7,2.8,6.7,2},
			{6.3,2.7,4.9,1.8},
			{6.7,3.3,5.7,2.1},
			{7.2,3.2,6,1.8},
			{6.2,2.8,4.8,1.8},
			{6.1,3,4.9,1.8},
			{6.4,2.8,5.6,2.1},
			{7.2,3,5.8,1.6},
			{7.4,2.8,6.1,1.9},
			{7.9,3.8,6.4,2},
			{6.4,2.8,5.6,2.2},
			{6.3,2.8,5.1,1.5},
			{6.1,2.6,5.6,1.4},
			{7.7,3,6.1,2.3},
			{6.3,3.4,5.6,2.4},
			{6.4,3.1,5.5,1.8},
			{6,3,4.8,1.8},
			{6.9,3.1,5.4,2.1},
			{6.7,3.1,5.6,2.4},
			{6.9,3.1,5.1,2.3},
			{5.8,2.7,5.1,1.9},
			{6.8,3.2,5.9,2.3},
			{6.7,3.3,5.7,2.5},
			{6.7,3,5.2,2.3},
			{6.3,2.5,5,1.9},
			{6.5,3,5.2,2},
			{6.2,3.4,5.4,2.3},
			{5.9,3,5.1,1.8},
		};
		
		auto means_clusters = dkm::kmeans_lloyd(iris_data, 3);
		auto means = std::get<0>(means_clusters);
		auto clusters = std::get<1>(means_clusters);
		// verify results
		EXPECT(means.size() == 3u);
		EXPECT(clusters.size() == iris_data.size());
		std::vector<std::array<float, 4>> expected_means{{5.006, 3.418, 1.464, 0.244}, {5.90161, 2.74839, 4.39355, 1.43387}, {6.85, 3.07368, 5.74211, 2.07105}};
		std::sort(means.begin(), means.end());
		// Check the means, with some allowance for error
		for (size_t i = 0; i < means.size(); ++i) {
			for (size_t j = 0; j < means[i].size(); ++j) {
				EXPECT(means[i][j] == lest::approx(expected_means[i][j]).epsilon(0.05));
			}
		}
		// Can't verify clusters easily because order may differ from run to run
		// Sorting the means before assigning clusters would help, but would also slow the algorithm down
		EXPECT(std::count(clusters.cbegin(), clusters.cend(), 0) > 0);
		EXPECT(std::count(clusters.cbegin(), clusters.cend(), 1) > 0);
		EXPECT(std::count(clusters.cbegin(), clusters.cend(), 2) > 0);
		EXPECT(std::count(clusters.cbegin(), clusters.cend(), 3) == 0);
	},
};

int main(int argc, char** argv) {
	return lest::run(specification, argc, argv);
}