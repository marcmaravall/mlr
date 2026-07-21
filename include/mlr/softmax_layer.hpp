#pragma once

#include "utils.hpp"
#include "layer.hpp"
#include "vector.hpp"

#include <cmath>
#include <math.h>

namespace mlr {

class softmax_layer : public layer {
private:
    vector m_last_output {};

public:
    softmax_layer() = default;
    ~softmax_layer() override = default;

private:


public:
    tensor forward(const tensor& input) override {
        vector res = utils::to_vector(input);
        
        double sum = 0.0;
        const int n = input.size();
        for (int i = 0; i < n; i++)
            sum += std::exp(input[i]);

        for (double& v : res) {
            v = std::exp(v)/sum;
        }
    
        m_last_output = res;
        return res;
    }

    tensor backward(const tensor& grad_output) override {
        vector grad = utils::to_vector(grad_output);

        const int n = grad.size();
        vector grad_input(n);

        double dot = 0.0;
        for (size_t i = 0; i < n; i++) {
            dot += grad[i]*m_last_output[i];
        }

        for (size_t i = 0; i < n; i++) {
            grad_input[i] = m_last_output[i]*(grad[i]-dot);
        }

        return grad_input;
    }

public:
    void update(double learning_rate) override {}
};

} // namespace mlr