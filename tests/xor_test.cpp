// MLR XOR TEST

#include <iostream>
#include <memory>
#include <vector>
#include <chrono>

#include <mlr/mlr.hpp>

int main() {
    mlr::sequential xor_model (
        std::make_unique<mlr::dense_layer>(2, 4),
        std::make_unique<mlr::sigmoid_layer>(),
        std::make_unique<mlr::dense_layer>(4, 1),
        std::make_unique<mlr::sigmoid_layer>()
    );

    std::vector<mlr::vector> inp_train = {
        {0.0, 0.0},
        {1.0, 0.0},
        {0.0, 1.0},
        {1.0, 1.0},
    };

    std::vector<mlr::vector> out_train = {
        {0.0},
        {1.0},
        {1.0},
        {0.0}
    };

    const double learning_rate = 0.1;
    const std::size_t epochs = 20000;

    std::cout << "MLR XOR TEST: \n";
    std::cout << "epochs: " << epochs << "\n";
    std::cout << "learning rate: " << learning_rate << "\n\n";

    auto start = std::chrono::high_resolution_clock::now();
    xor_model.train(inp_train, out_train, learning_rate, epochs);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = end-start;
    std::cout << "Model trained in " << duration.count()/1e9 << "s\n";

    for (int i = 0; i < inp_train.size(); i++) {
        mlr::vector res = xor_model(inp_train[i]);
        std::cout << inp_train[i][0] << " xor " << inp_train[i][1] << " = " << res[0] << "\n";
    }

    return 0;
}
