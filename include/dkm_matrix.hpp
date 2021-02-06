#pragma once

#ifndef DKM_MATRIX_HPP
#define DKM_MATRIX_HPP

#include <vector>
#include <cassert>
#include <algorithm>

namespace dkm
{
// This class is only an interface! Not designed to be used outside library internals.

template<typename T>
class as_matrix
{
    const T *data = nullptr;

    // column major indexer
    auto cm_indexer(size_t i, size_t j) const -> const T&
    {
        assert(i >= 0 && j >= 0 && i < n_rows && j < n_cols);
        return data[j * n_rows + i];
    }

     // row major indexer
    auto rm_indexer(size_t i, size_t j) const -> const T&
    {
        assert(i >= 0 && j >= 0 && i < n_rows && j < n_cols);
        return data[i * n_cols + j];
    }

    const T& (as_matrix<T>::*indexer) (size_t, size_t) const = nullptr;

public:
    const size_t n_rows, n_cols;

    as_matrix(const T *data, size_t n_rows, size_t n_cols, bool col_major = true)
        : data(data), n_rows(n_rows), n_cols(n_cols),
          indexer((col_major)? &as_matrix<T>::cm_indexer: &as_matrix<T>::rm_indexer)
    {}

    auto row(size_t i) const -> std::vector<T>;
    auto operator()(size_t i, size_t j) const -> const T&
    {
        return ((this)->*(this->indexer))(i, j);
    }
};


template<typename T>
auto as_matrix<T>::row(size_t i) const -> std::vector<T>
{
    auto res = std::vector<T>(n_cols);
    for(size_t j = 0; j < n_cols; j++)
    {
        res[j] = (*this)(i, j);
    }
    return res;
}

}
#endif
