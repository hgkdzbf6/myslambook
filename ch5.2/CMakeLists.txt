cmake_minimum_required(VERSION 2.9)
project(ch3_1)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Pangolin REQUIRED)
find_package(PCL REQUIRED COMPONENT common io)

add_compile_options("-std=c++11")
include_directories(
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${Pangolin_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
)
message(${EIGEN3_INCLUDE_DIR})
message(${OpenCV_INCLUDE_DIRS})
message(${Pangolin_INCLUDE_DIRS})

add_definitions( ${PCL_DEFINITIONS})
add_executable(hello
    src/main.cc
)
target_link_libraries(hello 
${EIGEN3_LIBS} 
${Pangolin_LIBRARIES} 
${OpenCV_LIBS} 
${PCL_LIBRARIES})