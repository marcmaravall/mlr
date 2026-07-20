#pragma once

#include "utils.hpp"
#include "layer.hpp"
#include "vector.hpp"

namespace mlr {

class relu_layer : public layer {
private:
    vector m_last_input {};

public:
    relu_layer() = default;
    ~relu_layer() override = default;

public:
    [[nodiscard]] constexpr static double relu(double x) { return (x > 0.0) ? x : 0.0; } 

public:
    tensor forward(const tensor& input) override {
        vector res = utils::to_vector(input);
        m_last_input = res;
        for (double& v : res) 
            v = relu(v);

        return res;
    }

    tensor backward(const tensor& grad_output) override {
        vector res = utils::to_vector(grad_output);
        for (std::size_t i = 0; i < res.size(); ++i)
            res[i] = (m_last_input(i) > 0.0) ? res[i] : 0.0;

        return res;
    }

    void update(double learning_rate) override {
        // nothing...
    }
};

} // namespace mlr