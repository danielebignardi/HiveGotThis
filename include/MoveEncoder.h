#ifndef MOVEENCODER_H
#define MOVEENCODER_H

#include "Board.h"
#include "Move.h"

#include <string>
#include <vector>

namespace HiveGotThis
{

constexpr int MoveFeatureDim = 32;

// Encode una mossa legale in un vettore stabile da 32 feature, visto dalla
// prospettiva del giocatore che deve muovere nella board passata.
std::vector<float> EncodeMoveFeatures(const Board& board, const Move& move);

std::string MoveFeaturesToJson(const std::vector<float>& features);

} // namespace HiveGotThis

#endif // MOVEENCODER_H
