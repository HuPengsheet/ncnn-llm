cmake_minimum_required(VERSION 3.10)
project(llm)
set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_PREFIX_PATH "/home/hp/code/llm/ncnn-20240410/build/install")
find_package(ncnn CONFIG)

add_executable(run)
target_sources(run PUBLIC src/main.cpp src/token.cpp src/llm.cpp)
target_include_directories(run PUBLIC include /home/hp/code/llm/ncnn-20240410/build/install/include/ncnn)
target_link_libraries(run PUBLIC ncnn )