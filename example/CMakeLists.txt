project(LinearAlgebra)

add_executable(matrix_handler_test)

target_sources(matrix_handler_test
    PRIVATE
    ${CMAKE_SOURCE_DIR}/src/matrix_handler.cpp
)

set_target_properties(matrix_handler_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/example"
)

target_link_libraries(matrix_handler_test PRIVATE LinearAlgebra::linear_algebra)
