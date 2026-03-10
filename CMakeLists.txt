cmake_minimum_required(VERSION 3.10)
project(HiveGotThis)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include_directories(include)

add_library(hive_core
    src/Board.cpp
    src/Enums.cpp
    src/Move.cpp
    src/Engine.cpp
)

file(GLOB TEST_SOURCES tests/*.cpp)
foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_link_libraries(${TEST_NAME} hive_core)
endforeach()