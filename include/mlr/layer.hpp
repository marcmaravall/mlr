#pragma once

#include "tensor.hpp"

namespace mlr {

class layer {
public:
    layer() = default;
    virtual ~layer() = default;

public:
    virtual tensor forward(const tensor& input) = 0;
    virtual tensor backward(const tensor& grad_output) = 0;

    virtual void update(double learning_rate) = 0;
};

} // namespace mlr