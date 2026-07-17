#pragma once

#include <array>
#include <algorithm>
#include <cassert>
#include <cmath>

#include "tensor.hpp"

namespace mlr {

template<std::size_t n>
class vector : public tensor<1, n> {
public:
    vector() = default;
    vector(std::initializer_list<double> init) : tensor<1, n>(init) {}
    
    ~vector() = default;

public:
    constexpr std::size_t size() const noexcept {
        return n;
    }

public:
    double& operator[](std::size_t i) {
        assert(i < n);
        return this->mem[i];
    }

    const double& operator[](std::size_t i) const {
        assert(i < n);
        return this->mem[i];
    }

    vector operator+(const vector& rhs) const {
        vector r;
        for (std::size_t i = 0; i < n; ++i)
            r[i] = this->mem[i] + rhs[i];
        return r;
    }

    vector operator-(const vector& rhs) const {
        vector r;
        for (std::size_t i = 0; i < n; ++i)
            r[i] = this->mem[i] - rhs[i];
        return r;
    }

    vector operator*(const vector& rhs) const {
        vector r;
        for (std::size_t i = 0; i < n; ++i)
            r[i] = this->mem[i] * rhs[i];
        return r;
    }

    vector operator*(double s) const {
        vector r;
        for (std::size_t i = 0; i < n; ++i)
            r[i] = this->mem[i] * s;
        return r;
    }

public:
    double dot(const vector& rhs) const {
        double r = 0.0;
        for (std::size_t i = 0; i < n; ++i)
            r += this->mem[i] * rhs[i];
        return r;
    }

    double norm() const {
        return std::sqrt(dot(*this));
    }
};

} // namespace mlr
