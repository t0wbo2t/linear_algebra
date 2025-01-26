project(LinearAlgebra)

add_library(linear_algebra STATIC matrix_handler.cpp)
add_library(LinearAlgebra::linear_algebra ALIAS linear_algebra)

target_include_directories(linear_algebra
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

