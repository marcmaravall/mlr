#pragma once

#include "layer.hpp"
#include "dense_layer.hpp"
#include "vector.hpp"
#include "tensor.hpp"

#include <initializer_list>

namespace mlr {

// TODO: change name 
class nnetwork {
private:
    // FIXME: convert to unique ptr
    std::vector<layer*> m_layers;

public:
    nnetwork() = default;
    ~nnetwork() {
        for (layer* l : m_layers)
            delete l;
    }

public:
    tensor forward(const tensor& input) {
        tensor out = input;
        for (auto& l : m_layers)
            out = l->forward(out);
        return out;
    }

    tensor backward(const tensor& grad_output) {
        tensor grad = grad_output;
        for (auto it = m_layers.rbegin(); it != m_layers.rend(); ++it)
            grad = (*it)->backward(grad);
        return grad;
    }

public:
    vector mse_grad(const vector& pred, const vector& expected) {
        vector g(pred.size());
        for (std::size_t i = 0; i < g.size(); ++i)
            g[i] = 2.0 * (pred[i] - expected[i]) / static_cast<double>(g.size());
        return g;
    }

    void train(const std::vector<vector>& xs, const std::vector<vector>& ys, double lr, std::size_t epochs) {
        for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
            for (std::size_t sample = 0; sample < xs.size(); ++sample) {
                tensor pred = forward(xs[sample]);
 
                vector y_pred(pred.shape()[0]);
                for (std::size_t i = 0; i < y_pred.size(); ++i)
                    y_pred[i] = pred.at({i});
 
                vector grad = mse_grad(y_pred, ys[sample]);
                backward(grad);

                // update
                for (auto& l : m_layers) {
                    if (auto* dense = dynamic_cast<dense_layer*>(l))
                        dense->update(lr);
                }
            }
        }
    }


public:
    layer* add(layer* layer) noexcept {
        m_layers.push_back(layer);
        return layer;
    }

    nnetwork(std::initializer_list<layer*> layers) {
        for (layer* l : layers) {
            m_layers.push_back((l));
        }
    }
};

} // namespace mlr
