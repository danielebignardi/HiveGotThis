#ifndef EVALUATION_H
#define EVALUATION_H

#include "Board.h"

#include <array>
#include <string>

namespace HiveGotThis
{

constexpr int GraphFeatureNodeCount = NumPieceNames; 
constexpr int GraphFeatureNodeDim = 38;
constexpr int GraphFeatureGlobalDim = 5;
constexpr int GraphFeatureMaxEdges = 1024; \\limite massimo di archi che il grafo può contenere.

enum class GraphEdgeType : int
{
    Adjacent = 0,
    StackedOn = 1,
    SameColor = 2
};

struct GraphFeatures
{
    std::array<std::array<float, GraphFeatureNodeDim>, GraphFeatureNodeCount> nodeFeatures{};
    std::array<int, GraphFeatureMaxEdges> edgeFrom{};
    std::array<int, GraphFeatureMaxEdges> edgeTo{};
    std::array<int, GraphFeatureMaxEdges> edgeType{};
    int numEdges = 0;
    std::array<float, GraphFeatureGlobalDim> globalFeatures{};
};

// Valuta la posizione della board dal punto di vista di `perspective`.
// Restituisce un valore in [0.0, 1.0]:
//   1.0 = vittoria certa per perspective
//   0.0 = sconfitta certa
//   0.5 = posizione pari
double EvaluateBoard(const Board& board, Color perspective);

// Valuta euristicamente una mossa (usato per l'ordinamento in MCTS).
// Restituisce un valore in [0.0, 1.0] (sigmoide applicata internamente).
// Punteggio piu' alto = mossa piu' promettente.
// board deve essere lo stato PRIMA della mossa.
double EvaluateMove(const Board& board, const Move& move, Color perspective);

// Estrae la rappresentazione graph-based con 38 feature per nodo.
GraphFeatures ExtractGraphFeatures(const Board& board);

// Serializza le feature in JSON per self-play/training.
std::string GraphFeaturesToJson(const GraphFeatures& features);

} // namespace HiveGotThis

#endif // EVALUATION_H
