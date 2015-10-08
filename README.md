# KM #
## A generic C++11 k-means clustering implementation ##

This is a k-means clustering algorithm written in C++, intended to be used as a header-only library. Requires C++11.

The algorithm is based on [Lloyds Algorithm](https://en.wikipedia.org/wiki/Lloyd%27s_algorithm) and uses the [kmeans++](https://en.wikipedia.org/wiki/K-means%2B%2B) initialization method.

The library is located in the `include` directory and may be used under the terms of the MIT license (see LICENSE.md). The tests in the `src` directory are also licensed under the MIT license, except for lest.hpp, which has its own license (src/LICENSE_1_0.txt), the Boost Software License.
