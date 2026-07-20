#pragma once

#include "utils.hpp"
#include "layer.hpp"
#include "vector.hpp"

#include <math.h>

namespace mlr {

class sigmoid_layer : public layer {
private:
    vector m_last_output {};
    ~sigmoid_layer() override = default;

private:
    static double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

public:
    tensor forward(const tensor& input) override {
        vector res = utils::to_vector(input);
        for (double& v : res) 
            v = sigmoid(v);
    
        m_last_output = res;
        return res;
    }

    tensor backward(const tensor& grad_output) override {
        vector res = utils::to_vector(grad_output);
        for (std::size_t i = 0; i < res.size(); ++i)
            res[i] = res[i] * m_last_output[i] * (1.0 - m_last_output[i]);

        return res;
    }

public:
    void update(double learning_rate) override {}
};

} // namespace mlr