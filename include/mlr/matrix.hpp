#pragma once

#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <vector>

#include "tensor.hpp"
#include "vector.hpp"

namespace mlr {

class matrix : public tensor {
public:
    matrix() = default;
    ~matrix() = default;

    matrix(std::size_t rows, std::size_t cols) : tensor({rows, cols}) {}
    matrix(std::size_t rows, std::size_t cols, double fill_value) : tensor({rows, cols}, fill_value) {}

    matrix(std::size_t rows, std::size_t cols, std::initializer_list<double> init) : tensor({rows, cols}, std::vector<double>(init)) {}

public:
    [[nodiscard]] std::size_t rows() const noexcept { return shape()[0]; }
    [[nodiscard]] std::size_t cols() const noexcept { return shape()[1]; }

public:
    double& operator()(std::size_t r, std::size_t c) noexcept { return (*this)[r * cols() + c]; }
    const double& operator()(std::size_t r, std::size_t c) const noexcept { return (*this)[r * cols() + c]; }

public:
    [[nodiscard]] static matrix zero(std::size_t rows, std::size_t cols) {
        return matrix(rows, cols, 0.0);
    }

    [[nodiscard]] static matrix identity(std::size_t n) {
        matrix result(n, n, 0.0);
        for (std::size_t i = 0; i < n; ++i)
            result(i, i) = 1.0;
        return result;
    }

public:
    matrix operator+(const matrix& rhs) const { matrix r = *this; r += rhs; return r; }
    matrix& operator+=(const matrix& rhs) { tensor::operator+=(rhs); return *this; }

    matrix operator-(const matrix& rhs) const { matrix r = *this; r -= rhs; return r; }
    matrix& operator-=(const matrix& rhs) { tensor::operator-=(rhs); return *this; }

public:
    matrix operator*(double scalar) const { matrix r = *this; r *= scalar; return r; }
    matrix& operator*=(double scalar) { tensor::operator*=(scalar); return *this; }

    matrix operator/(double scalar) const { matrix r = *this; r /= scalar; return r; }
    matrix& operator/=(double scalar) { tensor::operator/=(scalar); return *this; }

public:
    matrix operator*(const matrix& rhs) const {
        if (cols() != rhs.rows())
            throw std::invalid_argument("matrix::operator*: invalid dimensions!");

        matrix result(rows(), rhs.cols(), 0.0);
        for (std::size_t r = 0; r < rows(); ++r) {
            for (std::size_t c = 0; c < rhs.cols(); ++c) {
                double sum = 0.0;
                for (std::size_t k = 0; k < cols(); ++k)
                    sum += (*this)(r, k) * rhs(k, c);
                result(r, c) = sum;
            }
        }
        return result;
    }

    vector operator*(const vector& rhs) const {
        if (cols() != rhs.size())
            throw std::invalid_argument("matrix::operator*: dimensiones invalid dimensions!");

        vector result(rows());
        for (std::size_t r = 0; r < rows(); ++r) {
            double sum = 0.0;
            for (std::size_t c = 0; c < cols(); ++c)
                sum += (*this)(r, c) * rhs(c);
            result(r) = sum;
        }
        return result;
    }

public:
    [[nodiscard]] 
    vector row(std::size_t r) const {
        vector result(cols());
        for (std::size_t c = 0; c < cols(); ++c)
            result(c) = (*this)(r, c);
        return result;
    }

    [[nodiscard]] 
    vector col(std::size_t c) const {
        vector result(rows());
        for (std::size_t r = 0; r < rows(); ++r)
            result(r) = (*this)(r, c);
        return result;
    }

public:
    [[nodiscard]] 
    matrix transpose() const {
        matrix result(cols(), rows());
        for (std::size_t r = 0; r < rows(); ++r)
            for (std::size_t c = 0; c < cols(); ++c)
                result(c, r) = (*this)(r, c);
        return result;
    }
};

inline matrix operator*(double scalar, const matrix& m) { return m * scalar; }

} // namespace mlr
