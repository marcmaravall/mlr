#pragma once

#include "matrix.hpp"
#include "vector.hpp"
#include "tensor.hpp"

namespace mlr {

class utils {
public:
    // copies a tensor to vector
    [[nodiscard]] 
    static vector to_vector(const tensor& t, std::size_t expected_size, const char* who) {
        if (t.dimensions() != 1 || t.shape(0) != expected_size)
            throw std::invalid_argument(std::string(who) + ": expected a vector of size " + std::to_string(expected_size));
        
        vector res(expected_size);
        for (std::size_t i = 0; i < expected_size; ++i)
            res[i] = t[i];
        return res;
    }

    static void to_vector_into(vector& out, const tensor& t, std::size_t expected_size, const char* who) {
        if (t.dimensions() != 1 || t.shape(0) != expected_size)
            throw std::invalid_argument(std::string(who) + ": expected a vector of size " + std::to_string(expected_size));
    
        for (std::size_t i = 0; i < expected_size; ++i)
            out[i] = t[i];
    }

    [[nodiscard]]
    static vector to_vector(const tensor& t) {
        vector res(t.shape(0));
        for (std::size_t i = 0; i < res.size(); ++i)
            res[i] = t[i];
        return res;
    }
};

}