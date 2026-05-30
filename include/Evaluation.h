#ifndef EVALUATION_H
#define EVALUATION_H

#include "Board.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace HiveGotThis
{


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

} // namespace HiveGotThis

#endif // EVALUATION_H
