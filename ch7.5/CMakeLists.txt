cmake_minimum_required(VERSION 2.9)
project(ch3_1)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules)
set(CMAKE_BUILD_TYPE "Debug")

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package( CSparse )
find_package(G2O)
add_compile_options("-std=c++11")
include_directories(
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${CSPARSE_INCLUDE_DIR}
    ${G2O_INCLUDE_DIR}
    /usr/local/include/g2o/EXTERNAL/csparse
)

add_executable(hello
    src/main.cc
)
target_link_libraries(hello 
    ${EIGEN3_LIBS} 
    ${OpenCV_LIBS}
    ${CSPARSE_LIBRARY}
    ${G2O_CORE_LIBRARY} 
    ${G2O_STUFF_LIBRARY}
    g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension 
)