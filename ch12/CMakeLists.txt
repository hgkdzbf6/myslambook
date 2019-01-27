cmake_minimum_required(VERSION 2.9)
project(ch3_1)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules)
set(CMAKE_BUILD_TYPE "Debug")

find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(DBoW3 REQUIRED)
find_package(OpenCV REQUIRED)


add_compile_options("-std=c++11")
include_directories(
    ${EIGEN3_INCLUDE_DIR}
    ${Pangolin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${DBoW3_INCLUDE_DIRS}
)
message(${EIGEN3_INCLUDE_DIR})
message(${Pangolin_INCLUDE_DIRS})

add_executable(hello
    src/main.cc
)
target_link_libraries(hello 
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${DBoW3_LIBRARIES}
${OpenCV_LIBRARIES}
)