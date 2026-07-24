#pragma once

#include "layer.hpp"
#include "dense_layer.hpp"
#include "vector.hpp"
#include "tensor.hpp"
#include "utils.hpp"

#include <memory>
#include <initializer_list>
#include <iostream>

namespace mlr {

class sequential {
private:
    std::vector<std::unique_ptr<layer>> m_layers;

public:
    sequential() = default;
    template<typename... Layers>
    sequential(Layers&&... layers) {
        (m_layers.emplace_back(std::forward<Layers>(layers)), ...);
    }

    ~sequential() = default;

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

private:
    void update(double lr) {
        for (auto& l : m_layers)
            l->update(lr);
    }

public:
    vector mse_grad(const vector& pred, const vector& expected) {
        vector g(pred.size());
        mse_grad_into(g, pred, expected);
        return g;
    }

    void mse_grad_into(vector& g, const vector& pred, const vector& expected) {
        for (std::size_t i = 0; i < g.size(); ++i)
            g[i] = 2.0 * (pred[i] - expected[i]) / static_cast<double>(g.size());
    }

    double mse(const vector& pred, const vector& expected) {
        double res = 0;
        for (std::size_t i = 0; i < pred.size(); i++) {
            double v = pred[i]-expected[i];
            res += v*v;
        }

        return res / static_cast<double>(pred.size());
    }

    // TODO: set verbose to a bitflag for more options
    void train(const std::vector<vector>& xs, const std::vector<vector>& ys, double lr, std::size_t epochs, bool verbose = false) {
        if (xs.empty() || ys.empty())
            return;

        vector y_pred(ys[0].size());
        vector grad(y_pred.size());

        for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
            double epoch_loss = 0.0;
            for (std::size_t sample = 0; sample < xs.size(); ++sample) {
                tensor pred = forward(xs[sample]);
                for (std::size_t i = 0; i < y_pred.size(); ++i)
                    y_pred[i] = pred[i];

                epoch_loss += mse(y_pred, ys[sample]);
                mse_grad_into(grad, y_pred, ys[sample]);
                backward(grad);

                update(lr);
            }

            if (verbose && (epoch+1) % 100 == 0) {
                epoch_loss /= xs.size();
                std::cout << "[info] Epoch " << (epoch + 1) << "/" << epochs << " - Loss: " << epoch_loss << '\n';
            }
        }
    }

public:
    vector operator()(const vector& inp) {
        return utils::to_vector(forward(inp));
    }

public:
    void add(std::unique_ptr<layer> layer) noexcept {
        m_layers.push_back(std::move(layer));
    }
};

} // namespace mlr