#include "../include/matrix_handler.hpp"
#include <cassert>

int main() {
    std::cout << "Testing default constructor..." << '\n';
    MatrixHandler_2D<int> default_matrix;
    assert(default_matrix.count_rows() == 1);
    assert(default_matrix.count_columns() == 1);
    assert(default_matrix.get_element(0, 0) == 0);

    std::cout << "Testing parameterized constructor..." << '\n';
    MatrixHandler_2D<int> param_matrix(2, 3);
    assert(param_matrix.count_rows() == 2);
    assert(param_matrix.count_columns() == 3);

    std::cout << "Testing resizing functionality..." << '\n';
    assert(param_matrix.resize(3, 4));
    assert(param_matrix.count_rows() == 3);
    assert(param_matrix.count_columns() == 4);

    std::cout << "Testing element access and modification..." << '\n';
    param_matrix.set_element(1, 1, 42);
    assert(param_matrix.get_element(1, 1) == 42);

    try {
        param_matrix.get_element(5, 5);
        assert(false);
    } catch (const std::out_of_range& e) {
        std::cout << "Caught expected out-of-range exception: " << e.what() << '\n';
    }

    std::cout << "Testing equality operator..." << '\n';
    MatrixHandler_2D<int> equal_matrix(3, 4);
    equal_matrix.set_element(1, 1, 42);
    assert(equal_matrix == param_matrix);

    std::cout << "Testing matrix addition..." << '\n';
    equal_matrix.set_element(0, 0, 1);
    MatrixHandler_2D<int> sum_matrix = param_matrix + equal_matrix;
    assert(sum_matrix.get_element(0, 0) == 1);
    assert(sum_matrix.get_element(1, 1) == 84);

    std::cout << "Testing scalar addition..." << '\n';
    MatrixHandler_2D<int> scalar_sum = param_matrix + 10;
    assert(scalar_sum.get_element(1, 1) == 52);

    std::cout << "Testing matrix subtraction..." << '\n';
    MatrixHandler_2D<int> diff_matrix = param_matrix - equal_matrix;
    assert(diff_matrix.get_element(0, 0) == -1);
    assert(diff_matrix.get_element(1, 1) == 0);

    std::cout << "Testing matrix multiplication..." << '\n';
    MatrixHandler_2D<int> matrix1(2, 3, {{1, 2, 3}, {4, 5, 6}});
    MatrixHandler_2D<int> matrix2(3, 2, {{7, 8}, {9, 10}, {11, 12}});
    MatrixHandler_2D<int> product = matrix1 * matrix2;
    assert(product.count_rows() == 2);
    assert(product.count_columns() == 2);
    assert(product.get_element(0, 0) == 58);
    assert(product.get_element(1, 1) == 154);

    std::cout << "Testing scalar multiplication..." << '\n';
    MatrixHandler_2D<int> scalar_mul = 2 * matrix1;
    assert(scalar_mul.get_element(0, 0) == 2);
    assert(scalar_mul.get_element(1, 2) == 12);

    std::cout << "Testing edge cases..." << '\n';
    MatrixHandler_2D<int> empty_matrix(0, 0);
    assert(empty_matrix.count_rows() == 0);
    assert(empty_matrix.count_columns() == 0);

    try {
        empty_matrix.get_element(0, 0);
        assert(false);
    } catch (const std::out_of_range& e) {
        std::cout << "Caught expected exception for empty matrix: " << e.what() << '\n';
    }


    MatrixHandler_2D<float> test_det(2, 2, {{3.0, 7.0},{1.0, -4.0}});
    test_det.print();
    std::cout << "The determinant is: \n" << test_det.get_determinant() << '\n';

    test_det.set_inverse();
    test_det.print();

    MatrixHandler_2D<float> test_det_3d(3, 3, {{1.0, 2.0, 3.0},{0.0, 1.0, 4.0}, {0.0, 0.0, 1.0}});
    std::cout << "3x3 Vector..." << '\n';
    test_det_3d.print();

    test_det_3d.set_inverse();

    std::cout << "3x3 Vector Inverse..." << '\n';
    test_det_3d.print();
    std::cout << "All tests passed successfully!" << '\n';

    MatrixHandler_2D<float> slae(3, 3, {{1.0, 3.0, -1.0},{4.0, -1.0, 1.0}, {2.0, 4.0, 3.0}});
    std::vector<float> known = {13.0, 9.0,-6.0};
    auto result = slae.solve_slae(known);
    for(const auto num : result)
        std::cout << num << " ";
    std::cout << '\n';

    return 0;
}
