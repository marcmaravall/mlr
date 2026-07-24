#pragma once

#include <array>
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
    static constexpr std::size_t max_rank = 4;

    std::array<std::size_t, max_rank> m_shape {};
    std::array<std::size_t, max_rank> m_strides {};
    std::size_t m_rank = 0;
    std::vector<double> m_mem {};

private:
    void compute_strides() {
        if (m_rank == 0)
            return;
        m_strides[m_rank - 1] = 1;
        for (std::size_t i = m_rank - 1; i > 0; --i)
            m_strides[i - 1] = m_strides[i] * m_shape[i];
    }

    [[nodiscard]]
    std::size_t total_size() const noexcept {
        std::size_t total = 1;
        for (std::size_t i = 0; i < m_rank; ++i)
            total *= m_shape[i];
        return total;
    }

    void check_same_shape(const tensor& rhs) const {
        if (m_rank != rhs.m_rank ||
            !std::equal(m_shape.begin(), m_shape.begin() + m_rank, rhs.m_shape.begin()))
            throw std::invalid_argument("tensor: shape mismatch in elementwise operation");
    }

    void set_shape(std::initializer_list<std::size_t> shape) {
        if (shape.size() > max_rank)
            throw std::invalid_argument("tensor: rank > 4 not supported");
        m_rank = shape.size();
        std::copy(shape.begin(), shape.end(), m_shape.begin());
    }

    void set_shape(const std::vector<std::size_t>& shape) {
        if (shape.size() > max_rank)
            throw std::invalid_argument("tensor: rank > 4 not supported");
        m_rank = shape.size();
        std::copy(shape.begin(), shape.end(), m_shape.begin());
    }

public:
    tensor() = default;
    ~tensor() = default;

    tensor(const tensor& other) = default;
    tensor(tensor&& other) noexcept = default;
    tensor& operator=(const tensor& other) = default;
    tensor& operator=(tensor&& other) noexcept = default;

    explicit tensor(std::initializer_list<std::size_t> shape) {
        set_shape(shape);
        compute_strides();
        m_mem.resize(total_size());
    }

    explicit tensor(const std::vector<std::size_t>& shape) {
        set_shape(shape);
        compute_strides();
        m_mem.resize(total_size());
    }

    tensor(std::initializer_list<std::size_t> shape, double fill_value) {
        set_shape(shape);
        compute_strides();
        m_mem.resize(total_size(), fill_value);
    }

    tensor(const std::vector<std::size_t>& shape, double fill_value) {
        set_shape(shape);
        compute_strides();
        m_mem.resize(total_size(), fill_value);
    }

    tensor(std::initializer_list<std::size_t> shape, std::vector<double> data) {
        set_shape(shape);
        compute_strides();
        if (data.size() != total_size())
            throw std::invalid_argument("tensor: data size does not match shape");
        m_mem = std::move(data);
    }

    tensor(const std::vector<std::size_t>& shape, std::vector<double> data) {
        set_shape(shape);
        compute_strides();
        if (data.size() != total_size())
            throw std::invalid_argument("tensor: data size does not match shape");
        m_mem = std::move(data);
    }

public:
    [[nodiscard]] std::size_t dimensions() const noexcept { return m_rank; }
    [[nodiscard]] std::size_t size() const noexcept { return m_mem.size(); }
    [[nodiscard]] bool empty() const noexcept { return m_mem.empty(); }

    [[nodiscard]] const std::array<std::size_t, max_rank>& raw_shape() const noexcept { return m_shape; }
    [[nodiscard]] const std::array<std::size_t, max_rank>& raw_strides() const noexcept { return m_strides; }
    [[nodiscard]] std::size_t shape(std::size_t axis) const noexcept { return m_shape[axis]; }
    [[nodiscard]] std::size_t stride(std::size_t axis) const noexcept { return m_strides[axis]; }

    [[nodiscard]] double* data() noexcept { return m_mem.data(); }
    [[nodiscard]] const double* data() const noexcept { return m_mem.data(); }

    auto begin() noexcept { return m_mem.begin(); }
    auto end() noexcept { return m_mem.end(); }
    auto begin() const noexcept { return m_mem.begin(); }
    auto end() const noexcept { return m_mem.end(); }

public:
    double& at(const std::vector<std::size_t>& indices) {
        if (indices.size() != m_rank)
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

    void reshape(std::initializer_list<std::size_t> new_shape) {
        std::vector<std::size_t> tmp(new_shape);
        std::size_t total = 1;
        for (auto s : tmp) total *= s;
        if (total != m_mem.size())
            throw std::invalid_argument("tensor::reshape: element count mismatch");
        set_shape(new_shape);
        compute_strides();
    }

    void reshape(const std::vector<std::size_t>& new_shape) {
        std::size_t total = 1;
        for (auto s : new_shape) total *= s;
        if (total != m_mem.size())
            throw std::invalid_argument("tensor::reshape: element count mismatch");
        set_shape(new_shape);
        compute_strides();
    }

    [[nodiscard]] tensor clone() const {
        tensor result;
        result.m_shape = m_shape;
        result.m_strides = m_strides;
        result.m_rank = m_rank;
        result.m_mem = m_mem;
        return result;
    }

    bool operator==(const tensor& rhs) const noexcept {
        return m_rank == rhs.m_rank
            && std::equal(m_shape.begin(), m_shape.begin() + m_rank, rhs.m_shape.begin())
            && m_mem == rhs.m_mem;
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
