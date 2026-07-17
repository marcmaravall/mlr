#pragma once

#include <array>

namespace mlr {

template<std::size_t _dimensions, std::size_t _size>
class tensor {
private:

public:
    std::array<double, _size> mem {};

    constexpr std::size_t dimensions() const noexcept {
        return _dimensions;
    }

public:
    tensor() = default;
    tensor(std::initializer_list<double> init) : mem{} {
        std::copy_n(init.begin(), std::min(init.size(), this->mem.size()), this->mem.begin());
    }

    virtual ~tensor() = default;

};

} // namespace mlr