#pragma once

#include "layer.hpp"

namespace mlr {

class nnetwork {
private:
    std::vector<layer> m_layers;

public:
    nnetwork() = default;
    ~nnetwork() = default;
};

} // namespace mlr
