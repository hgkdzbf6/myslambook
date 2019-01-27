cmake_minimum_required(VERSION 2.9)
project(ch3_1)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package( G2O )
add_compile_options("-std=c++11")
include_directories(
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    # g2o/
)

add_executable(hello
    src/main.cc
)
target_link_libraries(hello 
    ${EIGEN3_LIBS} 
    ${OpenCV_LIBS} 
    ${G2O_CORE_LIBRARY} 
    ${G2O_STUFF_LIBRARY}
)