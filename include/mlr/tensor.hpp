#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace mlr {

class tensor {
private:
    std::vector<std::size_t> m_shape {};
    std::vector<std::size_t> m_strides {};
    std::vector<double> m_mem {};

private:
    void compute_strides() {
        m_strides.resize(m_shape.size());
        if (m_shape.empty())
            return;
        m_strides.back() = 1;
        for (std::size_t i = m_shape.size() - 1; i > 0; --i)
            m_strides[i - 1] = m_strides[i] * m_shape[i];
    }

    [[nodiscard]]
    static std::size_t total_size(const std::vector<std::size_t>& shape) {
        return std::accumulate(shape.begin(), shape.end(), std::size_t{1}, std::multiplies<>());
    }

    void check_same_shape(const tensor& rhs) const {
        if (m_shape != rhs.m_shape)
            throw std::invalid_argument("tensor: shape mismatch in elementwise operation");
    }

public:
    tensor() = default;
    ~tensor() = default;

    explicit tensor(std::vector<std::size_t> shape)
        : m_shape(std::move(shape)) {
        compute_strides();
        m_mem.resize(total_size(m_shape));
    }

    tensor(std::vector<std::size_t> shape, double fill_value)
        : m_shape(std::move(shape)) {
        compute_strides();
        m_mem.resize(total_size(m_shape), fill_value);
    }

    explicit tensor(std::initializer_list<std::size_t> shape) : tensor(std::vector<std::size_t>(shape)) {}

    tensor(std::vector<std::size_t> shape, std::vector<double> data)
        : m_shape(std::move(shape)) {
        compute_strides();
        if (data.size() != total_size(m_shape))
            throw std::invalid_argument("tensor: data size does not match shape");
        m_mem = std::move(data);
    }

public:
    [[nodiscard]] std::size_t dimensions() const noexcept { return m_shape.size(); }
    [[nodiscard]] std::size_t size() const noexcept { return m_mem.size(); }
    [[nodiscard]] bool empty() const noexcept { return m_mem.empty(); }
    [[nodiscard]] const std::vector<std::size_t>& shape() const noexcept { return m_shape; }
    [[nodiscard]] const std::vector<std::size_t>& strides() const noexcept { return m_strides; }
    [[nodiscard]] double* data() noexcept { return m_mem.data(); }
    [[nodiscard]] const double* data() const noexcept { return m_mem.data(); }

    auto begin() noexcept { return m_mem.begin(); }
    auto end() noexcept { return m_mem.end(); }
    auto begin() const noexcept { return m_mem.begin(); }
    auto end() const noexcept { return m_mem.end(); }

public:
    double& at(const std::vector<std::size_t>& indices) {
        if (indices.size() != m_shape.size())
            throw std::out_of_range("tensor::at: rank mismatch");

        std::size_t offset = 0;
        for (std::size_t axis = 0; axis < indices.size(); ++axis) {
            if (indices[axis] >= m_shape[axis])
                throw std::out_of_range("tensor::at: index out of bounds");
            offset += indices[axis] * m_strides[axis];
        }
        return m_mem[offset];
    }

    const double& at(const std::vector<std::size_t>& indices) const { return const_cast<tensor*>(this)->at(indices); }
    const double& at(std::initializer_list<std::size_t> indices) const { return at(std::vector<std::size_t>(indices)); }
    double& at(std::initializer_list<std::size_t> indices) { return at(std::vector<std::size_t>(indices)); }

public:
    double& operator[](std::size_t flat_index) noexcept { return m_mem[flat_index]; }
    const double& operator[](std::size_t flat_index) const noexcept { return m_mem[flat_index]; }

public:
    void fill(double value) noexcept {
        std::fill(m_mem.begin(), m_mem.end(), value);
    }

    void reshape(std::vector<std::size_t> new_shape) {
        if (total_size(new_shape) != m_mem.size())
            throw std::invalid_argument("tensor::reshape: element count mismatch");
        m_shape = std::move(new_shape);
        compute_strides();
    }

    [[nodiscard]] tensor clone() const {
        return tensor(m_shape, m_mem);
    }

    bool operator==(const tensor& rhs) const noexcept {
        return m_shape == rhs.m_shape && m_mem == rhs.m_mem;
    }

    bool operator!=(const tensor& rhs) const noexcept { 
        return !(*this == rhs); 
    }

    tensor& operator+=(const tensor& rhs) {
        check_same_shape(rhs);
        for (std::size_t i = 0; i < m_mem.size(); ++i) 
            m_mem[i] += rhs.m_mem[i];
        return *this;
    }

    tensor& operator-=(const tensor& rhs) {
        check_same_shape(rhs);
        for (std::size_t i = 0; i < m_mem.size(); ++i)
            m_mem[i] -= rhs.m_mem[i];
        return *this;
    }

    tensor& operator*=(const tensor& rhs) {
        check_same_shape(rhs);
        for (std::size_t i = 0; i < m_mem.size(); ++i) 
            m_mem[i] *= rhs.m_mem[i];
        return *this;
    }

    tensor& operator/=(const tensor& rhs) {
        check_same_shape(rhs);
        for (std::size_t i = 0; i < m_mem.size(); ++i) 
            m_mem[i] /= rhs.m_mem[i];
        return *this;
    }

    tensor& operator+=(double scalar) noexcept {
        for (auto& x : m_mem) 
            x += scalar;
        return *this;
    }

    tensor& operator-=(double scalar) noexcept {
        for (double& x : m_mem) 
            x -= scalar;
        return *this;
    }

    tensor& operator*=(double scalar) noexcept {
        for (double& x : m_mem) 
            x *= scalar;
        return *this;
    }

    tensor& operator/=(double scalar) {
        if (scalar == 0)
            throw std::runtime_error("tensor::operator/=: scalar is 0");

        for (double& x : m_mem) 
            x /= scalar;
        return *this;
    }

public:
    friend tensor operator+(tensor lhs, const tensor& rhs) { lhs += rhs; return lhs; }
    friend tensor operator-(tensor lhs, const tensor& rhs) { lhs -= rhs; return lhs; }
    friend tensor operator*(tensor lhs, const tensor& rhs) { lhs *= rhs; return lhs; }
    friend tensor operator/(tensor lhs, const tensor& rhs) { lhs /= rhs; return lhs; }

    friend tensor operator+(tensor lhs, double s) { lhs += s; return lhs; }
    friend tensor operator-(tensor lhs, double s) { lhs -= s; return lhs; }
    friend tensor operator*(tensor lhs, double s) { lhs *= s; return lhs; }
    friend tensor operator/(tensor lhs, double s) { lhs /= s; return lhs; }
    friend tensor operator*(double s, tensor rhs) { rhs *= s; return rhs; }

public:
    [[nodiscard]] double sum() const noexcept {
        return std::accumulate(m_mem.begin(), m_mem.end(), 0.0);
    }

    [[nodiscard]] double mean() const {
        assert(!m_mem.empty());
        return sum() / static_cast<double>(m_mem.size());
    }
};

} // namespace mlr