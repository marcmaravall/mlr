// MLR XOR TEST

#include <iostream>
#include <memory>
#include <vector>
#include <chrono>

#include <mlr/mlr.hpp>

int main() {
    mlr::sequential sum_model (
        std::make_unique<mlr::dense_layer>(2, 4, mlr::initialization::zero),
        std::make_unique<mlr::relu_layer>(),
        std::make_unique<mlr::dense_layer>(4, 1, mlr::initialization::zero)
    );

    std::vector<mlr::vector> inp_train = {
        {1.5, 4.5},
        {4.0, 6.0},
        {2.0, 3.0},
        {1.0, 8.0},
    };

    std::vector<mlr::vector> out_train = {
        {6.0},
        {10.0},
        {5.0},
        {9.0}
    };

    const double learning_rate = 0.01;
    const std::size_t epochs = 10000;

    std::cout << "MLR XOR TEST: \n";
    std::cout << "epochs: " << epochs << "\n";
    std::cout << "learning rate: " << learning_rate << "\n\n";

    auto start = std::chrono::high_resolution_clock::now();
    sum_model.train(inp_train, out_train, learning_rate, epochs, 1);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = end-start;
    std::cout << "Model trained in " << duration.count()/1e9 << "s\n";

    std::cout << "testcases:\n";
    std::vector<mlr::vector> test_inp = {
        {5.0, 5.0},
        {10, 150},
        {6.0, 18},
        {0.123, 0.934}
    };

    for (int i = 0; i < test_inp.size(); i++) {
        mlr::vector res = sum_model(test_inp[i]);
        std::cout << test_inp[i][0] << " + " << test_inp[i][1] << " = " << res[0] << "\n";
    }

    return 0;
}