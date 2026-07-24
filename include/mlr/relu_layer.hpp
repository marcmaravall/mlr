#pragma once

#include "utils.hpp"
#include "layer.hpp"
#include "vector.hpp"

namespace mlr {

class relu_layer : public layer {
private:
    vector m_last_input {};
    vector m_output_buffer {};
    vector m_grad_buffer {};

public:
    relu_layer() = default;
    ~relu_layer() override = default;

public:
    [[nodiscard]] constexpr static double relu(double x) { return (x > 0.0) ? x : 0.0; } 

public:
    tensor forward(const tensor& input) override {
        const std::size_t n = input.size();
        if (m_last_input.size() != n) {
            m_last_input = vector(n);
            m_output_buffer = vector(n);
        }

        for (std::size_t i = 0; i < n; ++i) {
            double v = input[i];
            m_last_input(i) = v;
            m_output_buffer(i) = relu(v);
        }

        return m_output_buffer;
    }

    tensor backward(const tensor& grad_output) override {
        const std::size_t n = grad_output.size();
        if (m_grad_buffer.size() != n)
            m_grad_buffer = vector(n);

        for (std::size_t i = 0; i < n; ++i)
            m_grad_buffer(i) = (m_last_input(i) > 0.0) ? grad_output[i] : 0.0;

        return m_grad_buffer;
    }

    void update(double learning_rate) override {
        // nothing...
    }
};

} // namespace mlr