# DKM #
## A generic C++11 k-means clustering implementation ##

This is a k-means clustering algorithm written in C++, intended to be used as a header-only library. Requires C++11.

The algorithm is based on [Lloyds Algorithm](https://en.wikipedia.org/wiki/Lloyd%27s_algorithm) and uses the [kmeans++](https://en.wikipedia.org/wiki/K-means%2B%2B) initialization method.

The library is located in the `include` directory and may be used under the terms of the MIT license (see LICENSE.md). The tests in the `src` directory are also licensed under the MIT license, except for lest.hpp, which has its own license (src/LICENSE_1_0.txt), the Boost Software License. The benchmarks located within the `bench` directory also fall under the MIT license. Benchmark data was obtained from the UCI Machine Learning Repository [here](https://archive.ics.uci.edu/ml/datasets/Iris) and [here](https://archive.ics.uci.edu/ml/datasets/US+Census+Data+(1990)).

A basic benchmark can be found in the bench folder. An example of the current results on an Intel i5-4210U:

```
OpenCV: 1.51998ms
DKM: 0.044276ms
```

This is only running k-means on a small data set (150 samples), and is only a single measurement, so do not interpret the results to mean that DKM is always faster than OpenCV.

Dependencies (bench):

 - OpenCV 2.4
