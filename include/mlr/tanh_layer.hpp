#pragma once

#include "utils.hpp"
#include "layer.hpp"
#include "vector.hpp"

#include <cmath>
#include <math.h>

namespace mlr {

class tanh_layer : public layer {
private:
    vector m_last_output {};

public:
    tanh_layer() = default;
    ~tanh_layer() override = default;

public:
    tensor forward(const tensor& input) override {
        vector res = utils::to_vector(input);
        for (double& v : res) 
            v = std::tanh(v);
    
        m_last_output = res;
        return res;
    }

    tensor backward(const tensor& grad_output) override {
        vector res = utils::to_vector(grad_output);
        for (std::size_t i = 0; i < res.size(); ++i)
            res[i] *= 1-(m_last_output[i]*m_last_output[i]);

        return res;
    }

public:
    void update(double learning_rate) override {}
};

} // namespace mlr