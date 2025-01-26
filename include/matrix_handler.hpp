#ifndef MATRIX_HANDLER_HPP
#define MATRIX_HANDLER_HPP

#include <vector>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

template <typename T>
class MatrixHandler_2D {
  private:
    size_t matrix_rows, matrix_columns;
    std::vector<std::vector<T>> matrix_2d;
    bool is_square() const;
    bool check_equality_r(const T& value_first, const T& value_second) const;
    void swap_row(const size_t row_first, const size_t row_second);

    void add_scaled_row(const size_t row_out, const size_t row_in, const T m_factor);
    void multiply_row(const size_t row_index, const T m_factor);

    bool join_column_wise(const MatrixHandler_2D<T>& another_matrix);
    size_t row_with_max_element(const size_t column_index, const size_t start_row) const;


  public:
    // Constructors
    MatrixHandler_2D();
    MatrixHandler_2D(size_t rows, size_t columns);
    MatrixHandler_2D(size_t rows, size_t columns, const std::vector<std::vector<T>>& input_data);
    MatrixHandler_2D(const MatrixHandler_2D<T>& input_matrix);

    // Utility Functions
    bool resize(size_t new_rows, size_t new_columns);

    T get_element(size_t row_index, size_t column_index) const;
    bool set_element(size_t row_index, size_t column_index, T value);
    size_t count_rows() const;
    size_t count_columns() const;
    void print() const;

    // Elementary Operations, Manipulators and Tranformers
    void transform_to_identity();
    bool set_inverse();
    std::vector<T> solve_slae(const std::vector<T>& known_values) const;

    bool separate(MatrixHandler_2D<T>& matrix_first, MatrixHandler_2D<T>& matrix_second, size_t column_index);
    MatrixHandler_2D<T> transpose() const;
    T get_determinant() const;

    // Overloader Operator
    bool operator== (const MatrixHandler_2D<T>& another_matrix) const;
    bool compare(const MatrixHandler_2D<T>& another_matrix, double tolerance) const;

    template <typename U>
      friend MatrixHandler_2D<U> operator+ (const MatrixHandler_2D<U>& lhs, const MatrixHandler_2D<U>& rhs);

    template <typename U>
      friend MatrixHandler_2D<U> operator+ (const U& lhs, const MatrixHandler_2D<U>& rhs);

    template <typename U>
      friend MatrixHandler_2D<U> operator+ (const MatrixHandler_2D<U>& lhs, const U& rhs);

    template <typename U>
      friend MatrixHandler_2D<U> operator- (const MatrixHandler_2D<U>& lhs, const MatrixHandler_2D<U>& rhs);
    
    template <typename U>
    friend MatrixHandler_2D<U> operator- (const U& lhs, const MatrixHandler_2D<U>& rhs);
    
    template <typename U>
    friend MatrixHandler_2D<U> operator- (const MatrixHandler_2D<U>& lhs, const U& rhs);

    template <typename U>
    friend MatrixHandler_2D<U> operator* (const MatrixHandler_2D<U>& lhs, const MatrixHandler_2D<U>& rhs);
    
    template <typename U>
    friend MatrixHandler_2D<U> operator* (const U& lhs, const MatrixHandler_2D<U>& rhs);
    
    template <typename U>
    friend MatrixHandler_2D<U> operator* (const MatrixHandler_2D<U>& lhs, const U& rhs);  
};

// Constructors
template <typename T>
MatrixHandler_2D<T>::MatrixHandler_2D()
  : matrix_rows(1),
    matrix_columns(1),
    matrix_2d(matrix_rows, std::vector<T>(matrix_columns, T())) {}

template <typename T>
MatrixHandler_2D<T>::MatrixHandler_2D(size_t rows, size_t columns)
  : matrix_rows(rows),
    matrix_columns(columns),
    matrix_2d(matrix_rows, std::vector<T>(matrix_columns, T())) {}

template <typename T>
MatrixHandler_2D<T>::MatrixHandler_2D(size_t rows, size_t columns, const std::vector<std::vector<T>>& input_data) 
  : matrix_rows(rows),
    matrix_columns(columns),
    matrix_2d(input_data) {}

template <typename T>
MatrixHandler_2D<T>::MatrixHandler_2D(const MatrixHandler_2D<T>& input_matrix)
  : matrix_rows(input_matrix.matrix_rows),
    matrix_columns(input_matrix.matrix_columns),
    matrix_2d(input_matrix.matrix_2d) {}

// Utility Functions
template <typename T>
bool MatrixHandler_2D<T>::resize(size_t new_rows, size_t new_columns) {
  try {
    matrix_2d.assign(new_rows, std::vector<T>(new_columns, T()));
    matrix_rows = new_rows;
    matrix_columns = new_columns;
    return true;
  } catch (const std::bad_alloc&) {
    return false;
  }
}

template <typename T>
T MatrixHandler_2D<T>::get_element(size_t row_index, size_t column_index) const {
    if (row_index >= matrix_rows || column_index >= matrix_columns)
        throw std::out_of_range("Index out of range.");
    return matrix_2d[row_index][column_index];
}

template <typename T>
bool MatrixHandler_2D<T>::set_element(size_t row_index, size_t column_index, T value) {
    if (row_index >= matrix_rows || column_index >= matrix_columns)
        return false;
    matrix_2d[row_index][column_index] = value;
    return true;
}

template <typename T>
size_t MatrixHandler_2D<T>::count_rows() const {
  return static_cast<size_t>(matrix_rows);
}

template <typename T>
size_t MatrixHandler_2D<T>::count_columns() const {
  return static_cast<size_t>(matrix_columns);
}

template <typename T>
void MatrixHandler_2D<T>::print() const {
  for (const auto& current_row : matrix_2d) {
    for (const auto& element : current_row) {
      std::cout << element << " ";
    }
    std::cout << '\n';
  }
}

// Manipulators and Tranformers
template <typename T>
bool MatrixHandler_2D<T>::is_square() const {
  return matrix_rows == matrix_columns;
}

template <typename T>
bool MatrixHandler_2D<T>::check_equality_r(const T& value_first, const T& value_second) const {
  return std::fabs(value_first - value_second) < std::numeric_limits<T>::epsilon();
}

template <typename T>
void MatrixHandler_2D<T>::swap_row(const size_t row_first, const size_t row_second) {
  if (row_first >= matrix_rows || row_second >= matrix_rows) {
    throw std::out_of_range("Row indices are out of range.\n");
  }
  std::swap(matrix_2d[row_first], matrix_2d[row_second]);
}

template <typename T>
void MatrixHandler_2D<T>::add_scaled_row(const size_t row_out, const size_t row_in, const T m_factor) {
  for(size_t column = 0; column < matrix_columns; ++column) {
    matrix_2d[row_out][column] += (matrix_2d[row_in][column] * m_factor);
  }
}

template <typename T>
void MatrixHandler_2D<T>::multiply_row(const size_t row_index, const T m_factor) {
  if (row_index >= matrix_rows) {
    throw std::out_of_range("Row index is out of bounds.");
  }

  for (auto& element : matrix_2d[row_index]) {
    element *= m_factor;
  }
}

template <typename T>
size_t MatrixHandler_2D<T>::row_with_max_element(const size_t column_index, const size_t start_row) const {
  if (start_row >= matrix_rows) {
    throw std::out_of_range("Start row index is out of bounds.\n");
  }

  if (matrix_2d.empty() || matrix_columns <= column_index) {
    throw std::out_of_range("Column index is out of bounds or matrix is empty.\n");
  }

  size_t row_with_max = start_row;
  T max_value = matrix_2d[row_with_max][column_index];

  // Iterate from start_row to the end of the matrix
  for (size_t row = start_row + 1; row < matrix_rows; ++row) {
      if (fabs(matrix_2d[row][column_index]) > fabs(max_value)) {
          max_value = matrix_2d[row][column_index];
          row_with_max = row;
      }
  }

  return row_with_max;
}

template <typename T>
void MatrixHandler_2D<T>::transform_to_identity() {
  if(!is_square()) {
    throw std::invalid_argument("Can not form an identity matrix that is not an square matrix.\n");
  }

  for(size_t row = 0; row < matrix_rows; ++row) {
    for(size_t column = 0; column < matrix_columns; ++column) {
      matrix_2d[row][column] = (row == column) ? T(1) : T(0);
    }
  }
}

template <typename T>
MatrixHandler_2D<T> MatrixHandler_2D<T>::transpose() const {
    MatrixHandler_2D<T> result(matrix_columns, matrix_rows);
    for (size_t row = 0; row < matrix_rows; ++row) {
        for (size_t column = 0; column < matrix_columns; ++column) {
            result.matrix_2d[column][row] = matrix_2d[row][column];
        }
    }
    return result;
}

template <typename T>
T MatrixHandler_2D<T>::get_determinant() const {
    if (!is_square()) {
        throw std::invalid_argument("Determinant is only defined for square matrices.");
    }

    MatrixHandler_2D<T> temp(*this);
    T determinant = T(1);

    for (size_t i = 0; i < matrix_rows; ++i) {
        size_t pivot = temp.row_with_max_element(i, i);

        if (std::fabs(temp.matrix_2d[pivot][i]) < std::numeric_limits<T>::epsilon()) {
            return T(0);
        }

        if (pivot != i) {
            temp.swap_row(i, pivot);
            determinant = -determinant;
        }

        T pivotValue = temp.matrix_2d[i][i];
        determinant *= pivotValue;

        temp.multiply_row(i, static_cast<T>(1) / pivotValue);

        for (size_t j = 0; j < matrix_rows; ++j) {
            if (j != i) {
                T factor = -temp.matrix_2d[j][i];
                temp.add_scaled_row(j, i, factor);
            }
        }
    }
    return determinant;
}

template <typename T>
bool MatrixHandler_2D<T>::set_inverse() {
  if (!is_square()) {
    throw std::invalid_argument("Only square matrices can have an inverse.");
  }

  size_t n = matrix_rows;

  MatrixHandler_2D<T> augmented_matrix(n, 2 * n);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      augmented_matrix.set_element(i, j, this->get_element(i, j));
    }
    augmented_matrix.set_element(i, n + i, static_cast<T>(1));
  }

  // Perform Gaussian-Jordan elimination
  for (size_t col = 0; col < n; ++col) {
    size_t max_row = augmented_matrix.row_with_max_element(col, col);
    if (std::fabs(augmented_matrix.get_element(max_row, col)) < std::numeric_limits<T>::epsilon()) {
      throw std::runtime_error("Matrix is singular and cannot be inverted.");
    }

    augmented_matrix.swap_row(col, max_row);

    T pivot = augmented_matrix.get_element(col, col);
    augmented_matrix.multiply_row(col, static_cast<T>(1) / pivot);

    for (size_t row = 0; row < n; ++row) {
      if (row != col) {
        T factor = -augmented_matrix.get_element(row, col);
        augmented_matrix.add_scaled_row(row, col, factor);
      }
    }
  }

  this->resize(n, n);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      this->set_element(i, j, augmented_matrix.get_element(i, n + j));
    }
  }

  return true;
}

template <typename T>
std::vector<T> MatrixHandler_2D<T>::solve_slae(const std::vector<T>& known_values) const {
  if (!is_square()) {
    throw std::invalid_argument("Matrix must be square to solve equations.");
  }

  size_t n = matrix_rows;

  // Create augmented matrix [A | b]
  MatrixHandler_2D<T> augmented_matrix(n, n + 1);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      augmented_matrix.set_element(i, j, this->get_element(i, j));
    }
    augmented_matrix.set_element(i, n, known_values.at(i));
  }

  // Gaussian-Jordan elimination
  for (size_t col = 0; col < n; ++col) {
    size_t max_row = augmented_matrix.row_with_max_element(col, col);
    if (std::fabs(augmented_matrix.get_element(max_row, col)) < std::numeric_limits<T>::epsilon()) {
      throw std::runtime_error("Matrix is singular or system has no unique solution.");
    }

    augmented_matrix.swap_row(col, max_row);

    T pivot = augmented_matrix.get_element(col, col);
    augmented_matrix.multiply_row(col, static_cast<T>(1) / pivot);

    for (size_t row = 0; row < n; ++row) {
      if (row != col) {
        T factor = -augmented_matrix.get_element(row, col);
        augmented_matrix.add_scaled_row(row, col, factor);
      }
    }
  }

  // Extract solution vector x
  std::vector<T> solution(n);
  for (size_t i = 0; i < n; ++i) {
    solution[i] = augmented_matrix.get_element(i, n);
  }

  return solution;
}

template <typename T>
bool MatrixHandler_2D<T>::join_column_wise(const MatrixHandler_2D<T>& another_matrix) {
  if (matrix_rows != another_matrix.matrix_rows) {
      return false; // Return false if the row count doesn't match
  }

  // Loop through each row and append the columns from another_matrix
  for (size_t row = 0; row < matrix_rows; ++row) {
      matrix_2d[row].insert(matrix_2d[row].end(), another_matrix.matrix_2d[row].begin(), another_matrix.matrix_2d[row].end());
  }

  matrix_columns += another_matrix.matrix_columns;

  return true; // Successfully joined the matrices
}

template <typename T>
bool MatrixHandler_2D<T>::separate(MatrixHandler_2D<T>& matrix_first, MatrixHandler_2D<T>& matrix_second, size_t column_index) {
  if(column_index >= matrix_columns) {
    throw std::invalid_argument("Please provide valid column index.\n");
    return false;
  }

  size_t new_rows = matrix_rows;
  size_t new_columns_first = column_index;
  size_t new_columns_second = matrix_columns - column_index;

  matrix_first.resize(matrix_rows, new_columns_first);
  matrix_second.resize(matrix_rows, new_columns_second);

  for(size_t row = 0; row < matrix_rows; ++row) {
    for(size_t column = 0; column < matrix_columns; ++column) {
      if(column < column_index) {
        matrix_first.set_element(row, column, this->get_element(row, column));
      } else {
        matrix_second.set_element(row, column - column_index, this->get_element(row, column));
      }
    }
  }
  return true;
}

template <typename T>
bool MatrixHandler_2D<T>::compare(const MatrixHandler_2D<T>& another_matrix, double tolerance) const {
  size_t another_matrix_rows = another_matrix.matrix_rows;
  size_t another_matrix_columns = another_matrix.matrix_columns;

  if((matrix_rows != another_matrix_rows) || (matrix_columns != another_matrix_columns)) {
    return false;
  }

  double c_sum = 0.0;
  for(size_t row = 0; row < matrix_rows; ++row) {
    for(size_t column = 0; column < matrix_columns; ++column) {
      T element_matrix = matrix_2d[row][column];
      T element_another_matrix = another_matrix.matrix_2d[row][column];

      c_sum += ((element_another_matrix - element_matrix) * (element_another_matrix - element_matrix));
    }

    double final_value = sqrt(c_sum / ((another_matrix_rows * another_matrix_columns) - 1));

    if(final_value < tolerance) {
      return true;
    } else {
      return false;
    }
  }
}

template <typename T>
bool MatrixHandler_2D<T>::operator==(const MatrixHandler_2D<T>& another_matrix) const {
    if(matrix_rows == another_matrix.matrix_rows && matrix_columns == another_matrix.matrix_columns &&
       matrix_2d == another_matrix.matrix_2d) {
       return true;
    }

    for(size_t row = 0; row < matrix_rows; ++row) {
      for(size_t column = 0; column < matrix_columns; ++column) {
        if(!check_equality_r(matrix_2d[row][column], another_matrix.matrix_2d[row][column])) {
            return false;
          }
      }
    }
    return true;
}

// Matrix Addition
template <typename U>
MatrixHandler_2D<U> operator+(const MatrixHandler_2D<U>& lhs, const MatrixHandler_2D<U>& rhs) {
    if (lhs.matrix_rows != rhs.matrix_rows || lhs.matrix_columns != rhs.matrix_columns)
        throw std::invalid_argument("Matrix dimensions must match for addition.");
    
    MatrixHandler_2D<U> result(lhs.matrix_rows, lhs.matrix_columns);
    for (size_t i = 0; i < lhs.matrix_rows; ++i)
        for (size_t j = 0; j < lhs.matrix_columns; ++j)
            result.matrix_2d[i][j] = lhs.matrix_2d[i][j] + rhs.matrix_2d[i][j];
    return result;
}

template <typename U>
MatrixHandler_2D<U> operator+ (const U& lhs, const MatrixHandler_2D<U>& rhs) {
  MatrixHandler_2D<U> result(rhs.matrix_rows, rhs.matrix_columns);
    for (size_t i = 0; i < result.matrix_rows; ++i)
        for (size_t j = 0; j < result.matrix_columns; ++j)
            result.matrix_2d[i][j] = lhs + rhs.matrix_2d[i][j];
    return result;
}

template <typename U>
MatrixHandler_2D<U> operator+ (const MatrixHandler_2D<U>& lhs, const U& rhs){
  MatrixHandler_2D<U> result(lhs.matrix_rows, lhs.matrix_columns);
    for (size_t i = 0; i < result.matrix_rows; ++i)
        for (size_t j = 0; j < result.matrix_columns; ++j)
            result.matrix_2d[i][j] = lhs.matrix_2d[i][j] + rhs;
    return result;
}

// Matrix Subtraction
template <typename U>
MatrixHandler_2D<U> operator-(const MatrixHandler_2D<U>& lhs, const MatrixHandler_2D<U>& rhs) {
    if (lhs.matrix_rows != rhs.matrix_rows || lhs.matrix_columns != rhs.matrix_columns)
        throw std::invalid_argument("Matrix dimensions must match for subtraction.");
    
    MatrixHandler_2D<U> result(lhs.matrix_rows, lhs.matrix_columns);
    for (size_t i = 0; i < lhs.matrix_rows; ++i)
        for (size_t j = 0; j < lhs.matrix_columns; ++j)
            result.matrix_2d[i][j] = lhs.matrix_2d[i][j] - rhs.matrix_2d[i][j];
    return result;
}

template <typename U>
MatrixHandler_2D<U> operator- (const U& lhs, const MatrixHandler_2D<U>& rhs) {
  MatrixHandler_2D<U> result(rhs.matrix_rows, rhs.matrix_columns);
    for (size_t i = 0; i < result.matrix_rows; ++i)
        for (size_t j = 0; j < result.matrix_columns; ++j)
            result.matrix_2d[i][j] = lhs - rhs.matrix_2d[i][j];
    return result;
}

template <typename U>
MatrixHandler_2D<U> operator- (const MatrixHandler_2D<U>& lhs, const U& rhs) {
  MatrixHandler_2D<U> result(lhs.matrix_rows, lhs.matrix_columns);
    for (size_t i = 0; i < result.matrix_rows; ++i)
        for (size_t j = 0; j < result.matrix_columns; ++j)
            result.matrix_2d[i][j] = lhs.matrix_2d[i][j] - rhs;
    return result;
}

// Matrix Multiplication
template <typename U>
MatrixHandler_2D<U> operator* (const MatrixHandler_2D<U>& lhs, const MatrixHandler_2D<U>& rhs) {
  if (lhs.matrix_columns != rhs.matrix_rows) {
    throw std::invalid_argument("Matrix dimensions must be compitable for multiplication.");
  }

  MatrixHandler_2D<U> result(lhs.matrix_rows, rhs.matrix_columns);
  for (size_t i = 0; i < lhs.matrix_rows; ++i)
    for (size_t j = 0; j < rhs.matrix_columns; ++j)
      for(size_t k = 0; k < lhs.matrix_columns; ++k)
            result.matrix_2d[i][j] += lhs.matrix_2d[i][k] * rhs.matrix_2d[k][j];
  return result;
}


template <typename U>
MatrixHandler_2D<U> operator* (const U& lhs, const MatrixHandler_2D<U>& rhs) {
  MatrixHandler_2D<U> result(rhs.matrix_rows, rhs.matrix_columns);
    for (size_t i = 0; i < result.matrix_rows; ++i)
        for (size_t j = 0; j < result.matrix_columns; ++j)
            result.matrix_2d[i][j] = lhs * rhs.matrix_2d[i][j];
    return result;
}

template <typename U>
MatrixHandler_2D<U> operator* (const MatrixHandler_2D<U>& lhs, const U& rhs) {
  MatrixHandler_2D<U> result(lhs.matrix_rows, lhs.matrix_columns);
    for (size_t i = 0; i < result.matrix_rows; ++i)
        for (size_t j = 0; j < result.matrix_columns; ++j)
            result.matrix_2d[i][j] = lhs.matrix_2d[i][j] * rhs;
    return result;
}

#endif //MATRIX_HANDLER_HPP
