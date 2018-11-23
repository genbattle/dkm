# DKM #
## A generic C++11 k-means clustering implementation ##

[![Build Status](https://travis-ci.org/genbattle/dkm.svg?branch=master)](https://travis-ci.org/genbattle/dkm)

This is a generic k-means clustering algorithm written in C++, intended to be used as a header-only library. Requires C++11.

The algorithm is based on [Lloyds Algorithm](https://en.wikipedia.org/wiki/Lloyd%27s_algorithm) and uses the [kmeans++](https://en.wikipedia.org/wiki/K-means%2B%2B) initialization method.

The library is located in the `include` directory and may be used under the terms of the MIT license (see LICENSE.md). The tests in the `src/test` directory are also licensed under the MIT license, except for `lest.hpp`, which has its own license (src/test/LICENSE_1_0.txt), the Boost Software License. The benchmarks located within the `bench` directory also fall under the MIT license. Benchmark data was obtained from the UCI Machine Learning Repository [here](https://archive.ics.uci.edu/ml/datasets/Iris) and the University of Eastern Finland [here](http://cs.joensuu.fi/sipu/datasets/).

`dkm.hpp` contains the standard serial implementation which depends only on C++11 support. `dkm_parallel.hpp` contains the parallel implementation which relies on OpenMP for acceleration; make sure to add `-fopenmp` (for GCC), `-fopenmp=libiomp5` (for Clang) or equivalent to your compiler flags to enable OpenMP if you use this implementation.

A simple benchmark can be found in the bench folder. An example of the current results on an Intel i5-4210U @ 1.7GHz:

```
# BEGINNING PROFILING #

## Dataset iris.data.csv ##
..............................
DKM: 0.0425224ms
DKM parallel: 0.0819724ms
OpenCV: 0.0532497ms

## Dataset s1.data.csv ##
..............................
DKM: 7.19246ms
DKM parallel: 4.95625ms
OpenCV: 4.56927ms

## Dataset birch3.data.csv ##
..............................
DKM: 5747.78ms
DKM parallel: 2985.93ms
OpenCV: 2367.55ms

## Dataset dim128.data.csv ##
....................
DKM: 36.7438ms
DKM parallel: 15.6721ms
OpenCV: ---

```

DKM is at least as fast as OpenCV for small datasets (< 500 points) and can handle any amount of dimensions (OpenCV is limited to 1D/2D data). At larger data sizes the parallel DKM implementation is significantly faster, and only slightly slower than the OpenCV implementation (in my testing).

### Benchmark Data Sets ###

`iris.data.csv` natural data taken from measurements of different iris plants. 150 points, 2 dimensions, 3 clusters. Source: [UCI machine learning repository](https://archive.ics.uci.edu/ml/datasets/Iris).

`s1.data.csv` synthetic data. 5000 points, 2 dimensions, 15 clusters. Source: P. Fränti and O. Virmajoki, "Iterative shrinking method for clustering problems", _Pattern Recognition_, 39 (5), 761-765, May 2006.

`birch3.data.csv` synthetic data large set. 100000 points, 2 dimensions, 100 clusters. Source: Zhang et al., "BIRCH: A new data clustering algorithm and its applications", _Data Mining and Knowledge Discovery_, 1 (2), 141-182, 1997

`dim128.data.csv` synthetic data with high dimensionality. 1024 points, 128 dimensions, 16 clusters. Source: P. Fränti, O. Virmajoki and V. Hautamäki, "Fast agglomerative clustering using a k-nearest neighbor graph", _IEEE Trans. on Pattern Analysis and Machine Intelligence_, 28 (11), 1875-1881, November 2006

### Usage ###

To use the DKM k-means implementation, simply include `include/dkm.hpp` and call `dkm::kmeans_lloyd()` with your data (`std::vector<std::array<>>`) and the number of cluster centers the algorithm should calculate for the data set. It is recommended that `float` or `double` are used for the input data types; with integer data types the algorithm may not converge.

Example:

```cpp
std::vector<std::array<float, 2>> data{{1.f, 1.f}, {2.f, 2.f}, {1200.f, 1200.f}, {2.f, 2.f}};
auto cluster_data = dkm::kmeans_lloyd(data, 2);
```

The parallel implementation works in the same way, except the header to include is `include/dkm_parallel.hpp` and the function to call is `dkm::kmeans_lloyd_parallel()`.

The return value of the `kmeans_lloyd` function is a `std::tuple<std::array<T, N>, std::vector<uint32_t>>` where the first element of the tuple is the cluster centroids (means) and the second element is a vector of indices that correspond to each of the input data elements. The indices returned in the second element of the tuple are cluster labels that map each corresponding element of the input data to a centroid in the first element of the tuple.

Printing the contents of the tuple for the example gives the following output:

```
Means:
	(1200,1200)
	(1.66667,1.66667)

Cluster labels:
	Point:         (1,1)         (2,2)   (1200,1200)         (2,2)
	Label:             1             1             0             1
```

We can see from the output that the means are at (1200, 1200) and (1.66667, 1.66667). The cluster labels show that the third data point is the only member of the first cluster. The first, second and fourth data points are members of the second cluster. The code used for this example is available in `src/example/main.cpp`.

### Building (tests and benchmarks) ###

For tests and benchmarks DKM uses a standard CMake out-of-tree build model.

To make everything:

```console
mkdir build
cd build
cmake .. && make
```

To build only the tests run `make dkm_tests` instead of `make`. To build only the benchmarks run `make dkm_bench`.

The tests can be run using the `make test` command or executing `./dkm_tests` in the build directory, and the benchmarks can likewise be run with `./dkm_bench`.


### Compatability ###

The following compilers are officially supported on Linux and tested via [Travis](https://travis-ci.org/genbattle/dkm):

- Clang 3.8
- Clang 5.0
- GCC 4.9
- GCC 7.0

GCC/Clang versions prior to the above are intentionally unsupported due to poor C++11 support. Other versions between or after the supported versions should work: if they don't, please create a bug report or pull request. Microsoft VC++ 12.0+ (Visual Studio 2013 or later) is intended to be supported, but does not currently have a CI build set up.

### Dependencies (test) ###

- CMake

### Dependencies (bench) ###

- OpenCV 2.4+
- OpenMP 3.1+
- CMake
