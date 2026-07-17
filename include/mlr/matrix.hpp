#pragma once

#include <array>
#include <algorithm>
#include <initializer_list>
#include <cstddef>

#include "vector.hpp"
#include "tensor.hpp"

namespace mlr {

template<std::size_t rows, std::size_t cols>
class matrix : public tensor<2, rows*cols> {
public:
    matrix() = default;
    matrix(std::initializer_list<double> init) : tensor<2, rows*cols>(init) {}

    ~matrix() = default;

public:
    constexpr std::size_t width() const noexcept {
        return cols;
    }

    constexpr std::size_t height() const noexcept {
        return rows;
    }

public:
    constexpr double& operator()(std::size_t row, std::size_t col) noexcept {
        return this->mem[row * cols + col];
    }

    constexpr const double& operator()(std::size_t row, std::size_t col) const noexcept {
        return this->mem[row * cols + col];
    }

public:
    matrix operator+(const matrix& rhs) const {
        matrix result;

        for (std::size_t r = 0; r < rows; ++r) {
            for (std::size_t c = 0; c < cols; ++c) {
                result(r, c) = (*this)(r, c) + rhs(r, c);
            }
        }

        return result;
    }

    matrix& operator+=(const matrix& rhs) {
        for (std::size_t r = 0; r < rows; ++r) {
            for (std::size_t c = 0; c < cols; ++c) {
                (*this)(r, c) += rhs(r, c);
            }
        }

        return *this;
    }

    matrix operator-(const matrix& rhs) const {
        matrix result;

        for (std::size_t r = 0; r < rows; ++r) {
            for (std::size_t c = 0; c < cols; ++c) {
                result(r, c) = (*this)(r, c) - rhs(r, c);
            }
        }

        return result;
    }

    matrix operator*(double scalar) const {
        matrix result = *this;

        for (auto& value : result.mem)
            value *= scalar;

        return result;
    }

    template<std::size_t other_cols>
    matrix<rows, other_cols> operator*(const matrix<cols, other_cols>& rhs) const {
        matrix<rows, other_cols> result;

        for (std::size_t r = 0; r < rows; ++r) {
            for (std::size_t c = 0; c < other_cols; ++c) {

                double sum = 0.0;

                for (std::size_t k = 0; k < cols; ++k) {
                    sum += (*this)(r, k) * rhs(k, c);
                }

                result(r, c) = sum;
            }
        }

        return result;
    }

    vector<rows> operator*(const vector<cols>& rhs) const {
        vector<rows> result;

        for (std::size_t r = 0; r < rows; ++r) {
            double sum = 0.0;

            for (std::size_t c = 0; c < cols; ++c) {
                sum += (*this)(r, c) * rhs[c];
            }

            result[r] = sum;
        }

        return result;
    }

    matrix<cols, rows> transpose() const {
        matrix<cols, rows> result;

        for (std::size_t r = 0; r < rows; ++r) {
            for (std::size_t c = 0; c < cols; ++c) {
                result(c, r) = (*this)(r, c);
            }
        }

        return result;
    }
};

template<std::size_t rows, std::size_t cols>
matrix<rows, cols> operator*(double scalar, const matrix<rows, cols>& mat) {
    return mat * scalar;
}

} // namespace mlr
