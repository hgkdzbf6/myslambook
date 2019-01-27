cmake_minimum_required(VERSION 2.9)
project(ch3_1)

find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)

add_compile_options("-std=c++11")
include_directories(
    ${EIGEN3_INCLUDE_DIR}
    Sophus/
    ${Pangolin_INCLUDE_DIRS}
)
message(${EIGEN3_INCLUDE_DIR})
message(${Pangolin_INCLUDE_DIRS})

add_executable(hello
    src/main.cc
)
target_link_libraries(hello 
${EIGEN3_LIBS} 
${Pangolin_LIBRARIES} )