cmake_minimum_required(VERSION 3.10)
project(HiveGotThis)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include_directories(include)

# Value network: richiede libtorch obbligatoriamente.
# Punta CMAKE_PREFIX_PATH alla cartella di libtorch, es.:
#   cmake -B build -S . -DCMAKE_PREFIX_PATH=third_party/libtorch
find_package(Torch REQUIRED)

# Libreria comune
add_library(hive_core
    src/Board.cpp
    src/Enums.cpp
    src/Move.cpp
    src/Engine.cpp
    src/Evaluation.cpp
    src/MCTS.cpp
    src/BoardEncoder.cpp
    src/NeuralEvaluator.cpp
)
target_include_directories(hive_core PUBLIC ${TORCH_INCLUDE_DIRS})
target_link_libraries(hive_core PUBLIC ${TORCH_LIBRARIES})
set_property(TARGET hive_core PROPERTY CXX_STANDARD 17)

# Eseguibile principale (il bot UHP)
add_executable(HiveEngine src/main.cpp)
target_link_libraries(HiveEngine hive_core)

# Test (escluso TestMoves: usa API Unix non disponibili su Windows)
foreach(test TestBoard TestSenzaBoard TestIsOneHive TestZobristHash TestZobristHash2 TestEngine TestIsPinned TestEncoderShapes TestEdgeAlignment TestPerft TestPerftMidgame TestMoves)
    add_executable(${test} tests/${test}.cpp)
    target_link_libraries(${test} hive_core)
endforeach()