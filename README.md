# Matrix Handler Library

Matrix Handler is a modular and template-based C++ library for performing various matrix operations, including:
- Basic matrix operations (addition, subtraction, multiplication, etc.)
- Row operations
- Determinant calculation
- Matrix inversion
- Solving systems of linear equations

This project is designed with flexibility and reusability in mind, utilizing modern C++ features (C++17 or above) and adhering to best practices for modularity and build automation using CMake.

---

## Features
- **Template-based Design:** Generic implementation of matrix operations using templates for compile-time type safety.
- **Modularity:** Clean separation of functionality into header and source files.
- **CMake Support:** Multi-directory CMake structure for easy build and integration.
- **Scalability:** Designed to handle small to large-scale matrix operations efficiently.

---

## Directory Structure
```
.
├── CMakeLists.txt             # Parent CMake file
├── example
│   └── CMakeLists.txt         # Example-specific CMake file
├── include
│   ├── CMakeLists.txt         # Include-specific CMake file
│   └── matrix_handler.hpp     # Header file with template implementations
├── LICENSE                    # License information
├── README.md                  # Project description
└── src
    ├── CMakeLists.txt         # Source-specific CMake file
    └── matrix_handler.cpp     # Implementation and testing of matrix operations
```

---

## Getting Started
### Prerequisites
Ensure you have the following installed:
- A C++17-compatible compiler (e.g., GCC, Clang, MSVC)
- CMake 3.15 or newer

### Building the Library
1. Clone the repository:
   ```bash
   git clone https://github.com/t0wbo2t/linear_algebra.git
   cd linear_algebra
   ```
2. Create a build directory and configure the project:
   ```bash
   mkdir build && cd build
   cmake ..
   ```
3. Build the project:
   ```bash
   cmake --build .
   ```

### Running Examples
You can find example usage of the library in the `example` directory. To run the examples:
```bash
cd build/example
./matrix_handler_test
```

---

## Usage
To use the library in your project, include the `matrix_handler.hpp` header and link the library:

### Example
```cpp
#include "matrix_handler.hpp"

int main() {
    MatrixHandler_2D<float> mat1(2, 2, {{3.0, 7.0},{1.0, -4.0}});
    MatrixHandler_2D<float> mat2(2, 2, {{3.0, 7.0},{1.0, -4.0}});

    auto result = mat1 + mat2;
    result.print();

    auto determinant = mat1.get_determinant();
    std::cout << "Determinant: " << determinant << std::endl;

    return 0;
}
```

---

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.

### Steps to Contribute
1. Fork the repository
2. Create a new branch for your feature/bugfix
3. Commit your changes
4. Push to your fork and create a pull request

---

## License
This project is licensed under the GPL-3.0 License. See the [LICENSE](LICENSE) file for details.

