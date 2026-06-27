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
    src/Evaluation.cpp
    src/MCTS.cpp
    src/BoardEncoder.cpp
)

# Eseguibile principale (il bot UHP)
add_executable(HiveEngine src/main.cpp)
target_link_libraries(HiveEngine hive_core)

# Test (escluso TestMoves: usa API Unix non disponibili su Windows)
foreach(test TestBoard TestSenzaBoard TestIsOneHive TestZobristHash TestZobristHash2 TestEngine TestIsPinned TestEncoderShapes TestEdgeAlignment TestPerft)
    add_executable(${test} tests/${test}.cpp)
    target_link_libraries(${test} hive_core)
endforeach()