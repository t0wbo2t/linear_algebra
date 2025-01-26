cmake_minimum_required(VERSION 3.30)

project(LinearAlgebra
        VERSION 0.1.0
        DESCRIPTION "A Modern C++ Linear Algebra Implementation"
        LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(src)
add_subdirectory(include)
add_subdirectory(example)
