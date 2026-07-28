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

// Indice del pezzo tra i nodi del grafo di BoardEncoder: i nodi sono i pezzi
// in gioco, nell'ordine dell'enum PieceName (stesso contratto di encode()).
// -1 se il pezzo non e' sulla board (piazzamento) o non e' valido.
int GraphNodeIndex(const Board& board, PieceName piece);

// Frammento JSON '"src":K,"dst":[[nodo,slot],...]' che descrive la mossa in
// termini di nodi del grafo, per una futura policy sugli embedding dei nodi:
//   src = nodo del pezzo mosso (-1 = piazzamento o pass);
//   dst = vicini occupati della destinazione, coppie [nodo, slot direzione
//         dal vicino VERSO la destinazione]; slot 6 (up) = la destinazione
//         e' in cima a quel pezzo (salita).
// Il pezzo mosso non compare mai in dst: dopo la mossa non e' piu' nella
// cella di partenza (se sotto ha un altro pezzo, vale quello).
std::string MoveStructuralJson(const Board& board, const Move& move);

} // namespace HiveGotThis

#endif // MOVEENCODER_H
