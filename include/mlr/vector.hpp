#pragma once

#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <vector>

#include "tensor.hpp"

namespace mlr {

class vector : public tensor {
public:
    vector() = default;

    explicit vector(std::size_t n) : tensor({n}) {}

    vector(std::size_t n, double fill_value) : tensor({n}, fill_value) {}

    vector(std::initializer_list<double> init) : tensor({init.size()}, std::vector<double>(init)) {}

public:
    [[nodiscard]] std::size_t length() const noexcept { return size(); }

public:
    double& operator()(std::size_t i) { return (*this)[i]; }
    const double& operator()(std::size_t i) const { return (*this)[i]; }

public:
    [[nodiscard]] double dot(const vector& other) const {
        if (size() != other.size())
            throw std::invalid_argument("vector::dot: invalid dimensions!");
        double sum = 0.0;
        for (std::size_t i = 0; i < size(); ++i)
            sum += (*this)[i] * other[i];
        return sum;
    }

    [[nodiscard]] double norm() const { return std::sqrt(dot(*this)); }

    [[nodiscard]] vector normalized() const {
        const double n = norm();
        vector result = *this;
        for (auto& x : result) x /= n;
        return result;
    }

public:
    vector operator+(const vector& rhs) const { vector r = *this; r += rhs; return r; }
    vector& operator+=(const vector& rhs) { tensor::operator+=(rhs); return *this; }

    vector operator-(const vector& rhs) const { vector r = *this; r -= rhs; return r; }
    vector& operator-=(const vector& rhs) { tensor::operator-=(rhs); return *this; }

    vector operator*(double scalar) const { vector r = *this; r *= scalar; return r; }
    vector& operator*=(double scalar) { tensor::operator*=(scalar); return *this; }

    vector operator/(double scalar) const { vector r = *this; r /= scalar; return r; }
    vector& operator/=(double scalar) { tensor::operator/=(scalar); return *this; }
};

inline vector operator*(double scalar, const vector& v) { return v * scalar; }

} // namespace mlr
