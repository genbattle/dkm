#include "../../include/dkm.hpp"
#include <iostream>
#include <sstream>
#include <iomanip>

int main() {
	std::vector<std::array<float, 2>> data{{1.f, 1.f}, {2.f, 2.f}, {1200.f, 1200.f}, {2.f, 2.f}};
	auto cluster_data = dkm::kmeans_lloyd(data, 2);

	std::cout << "Means:" << std::endl;
	for (const auto& mean : std::get<0>(cluster_data)) {
		std::cout << "\t(" << mean[0] << "," << mean[1] << ")" << std::endl;
	}
	std::cout << "\nCluster labels:" << std::endl;
	std::cout << "\tPoint:";
	for (const auto& point : data) {
		std::stringstream value;
		value << "(" << point[0] << "," << point[1] << ")";
		std::cout << std::setw(14) << value.str();
	}
	std::cout << std::endl;
	std::cout << "\tLabel:";
	for (const auto& label : std::get<1>(cluster_data)) {
		std::cout << std::setw(14) << label;
	}
	std::cout << std::endl;
}
