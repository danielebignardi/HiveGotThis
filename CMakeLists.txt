cmake_minimum_required(VERSION 3.10)
project(HiveGotThis)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include_directories(include)

# Libreria comune
add_library(hive_core
    src/Board.cpp
    src/Enums.cpp
    src/Move.cpp
    src/Engine.cpp
)

# Un eseguibile per ogni file di test
foreach(test TestBoard TestSenzaBoard TestIsOneHive TestZobristHash TestZobristHash2 TestEngine)
    add_executable(${test} tests/${test}.cpp)
    target_link_libraries(${test} hive_core)
endforeach()

add_executable(main src/main.cpp)
target_link_libraries(main hive_core)