// clang-format disabled because clang-format doesn't format lest's macros correctly
// clang-format off
/*
Test cases for dkm.hpp

This is just simple test harness without any external dependencies.
*/

#include "../../include/dkm.hpp"
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
};

int main(int argc, char** argv) {
	return lest::run(specification, argc, argv);
}