#pragma once

#include "layer.hpp"
#include "matrix.hpp"
#include "vector.hpp"
#include "utils.hpp"

#include <cmath>
#include <random>
#include <stdexcept>
#include <iostream>

namespace mlr {

enum class initialization {
    zero,
    xavier,
    kaiming_normal,
    kaiming_uniform,
    orthogonal,
    // load_file
};

class dense_layer : public layer {
private:
    // input vector size
    std::size_t m_in_features = 0;
    // output vector size
    std::size_t m_out_features = 0;
    bool m_initialized = false;

private:
    matrix m_weights {};
    vector m_bias {};

    // backpropagation:
    matrix m_grad_weights {};
    vector m_grad_bias {};
    vector m_last_input {};

private:
    void zero_init() {
        m_weights = matrix(m_out_features, m_in_features, 0.0);
        m_bias = vector(m_out_features, 0.0);
    }

    // TODO: merge the two functions and only change distribution
    void kaiming_normal_init(unsigned seed = std::random_device{}()) {
        m_weights = matrix(m_out_features, m_in_features);
        m_bias = vector(m_out_features, 0.0);

        double bound = std::sqrt(2.0 / static_cast<double>(m_in_features));
        std::mt19937 rng(seed);
        std::normal_distribution<double> dist(0, bound);

        for (std::size_t o = 0; o < m_out_features; ++o)
            for (std::size_t i = 0; i < m_in_features; ++i)
                m_weights(o, i) = dist(rng);
    }

    void kaiming_uniform_init(unsigned seed = std::random_device{}()) {
        m_weights = matrix(m_out_features, m_in_features);
        m_bias = vector(m_out_features, 0.0);

        double bound = std::sqrt(6.0 / static_cast<double>(m_in_features));
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> dist(-bound, bound);

        for (std::size_t o = 0; o < m_out_features; ++o)
            for (std::size_t i = 0; i < m_in_features; ++i)
                m_weights(o, i) = dist(rng);
    }

    // TODO: implement
    void orthogonal_init() {
        throw new std::runtime_error("dense_layer::orthogonal_init is not implemented!");
    }

    void xavier_init(unsigned seed = std::random_device{}()) {
        m_weights = matrix(m_out_features, m_in_features);
        m_bias = vector(m_out_features, 0.0);

        // xavier initializarion:   
        // https://www.geeksforgeeks.org/deep-learning/xavier-initialization/
        const double limit = std::sqrt(6.0 / static_cast<double>(m_in_features + m_out_features));
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> dist(-limit, limit);

        for (std::size_t o = 0; o < m_out_features; ++o)
            for (std::size_t i = 0; i < m_in_features; ++i)
                m_weights(o, i) = dist(rng);
    }

    void init_parameters(initialization init = initialization::xavier) {
        m_initialized = true;
        switch(init) {
        case initialization::xavier: 
            xavier_init(); 
            break;
        case initialization::zero:
            zero_init();
            break;
        case initialization::kaiming_normal:
            kaiming_normal_init();
            break;
        case initialization::kaiming_uniform:
            kaiming_uniform_init();
            break;
        case initialization::orthogonal:
            orthogonal_init();
            break;

        default:    // default initialiation:
            zero_init();
        }

        // gradients are always initialized to 0
        m_grad_weights = matrix(m_out_features, m_in_features, 0.0);
        m_grad_bias = vector(m_out_features, 0.0);
    }

public:
    dense_layer() = default;
    ~dense_layer() noexcept override = default;

    dense_layer(std::size_t in_features, std::size_t out_features, initialization init = initialization::xavier)
        : m_in_features(in_features), m_out_features(out_features) {
        if (in_features == 0 || out_features == 0)
            throw std::invalid_argument("dense_layer: in_features and out_features must be > 0");
        init_parameters(initialization::xavier);
    }

public:
    tensor forward(const tensor& input) override {
        if (!m_initialized)
            throw std::runtime_error("dense_layer::forward: layer not initialized (use the sized constructor)");
        const vector inp = utils::to_vector(input, m_in_features, "dense_layer::forward");
        
        vector res = m_weights*inp + m_bias;
        m_last_input = inp;
        return res;
    }

    tensor backward(const tensor& grad_output) override {
        if (!m_initialized)
            throw std::runtime_error("dense_layer::backward: layer not initialized (use the sized constructor)");
        if (m_last_input.empty())
            throw std::runtime_error("dense_layer::backward: forward() must be called before backward()");

        const vector g = utils::to_vector(grad_output, m_out_features, "dense_layer::backward");

        m_grad_bias = g;

        m_grad_weights = matrix(m_out_features, m_in_features);
        for (std::size_t o = 0; o < m_out_features; ++o)
            for (std::size_t i = 0; i < m_in_features; ++i)
                m_grad_weights(o, i) = g(o) * m_last_input(i);

        vector grad_input = m_weights.transpose() * g;
        return grad_input;
    }

public:
    void update(double learning_rate) override {
        m_weights -= learning_rate * m_grad_weights;
        m_bias -= learning_rate * m_grad_bias;
    }

public:
    [[nodiscard]] const matrix& weights() const noexcept { return m_weights; }
    [[nodiscard]] const vector& bias() const noexcept { return m_bias; }
    [[nodiscard]] const matrix& grad_weights() const noexcept { return m_grad_weights; }
    [[nodiscard]] const vector& grad_bias() const noexcept { return m_grad_bias; }
};

} // namespace mlr
