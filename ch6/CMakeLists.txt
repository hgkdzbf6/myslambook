cmake_minimum_required(VERSION 2.9)
project(ch3_1)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)

add_compile_options("-std=c++11")
include_directories(
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    /usr/local/include/cere
)
message(${EIGEN3_INCLUDE_DIR})
message(${CERES_LIBRARIES})

add_definitions( ${PCL_DEFINITIONS})
add_executable(hello
    src/main.cc
)
target_link_libraries(hello 
${EIGEN3_LIBS} 
${OpenCV_LIBS} 
${CERES_LIBRARIES}
)