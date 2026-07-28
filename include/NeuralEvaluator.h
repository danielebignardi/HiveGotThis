#ifndef NEURALEVALUATOR_H
#define NEURALEVALUATOR_H

#include "Board.h"
#include "BoardEncoder.h"
#include "Move.h"

#include <torch/script.h>

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace HiveGotThis
{

// TorchScriptValueEvaluator e' il lato C++ del ponte Python -> C++.
//
// Python addestra la rete di valore e la esporta come file TorchScript:
//
//     hive_value_gnn.pt
//
// Questa classe carica quel file una sola volta, poi esegue l'inferenza
// direttamente in C++ tramite libtorch. Durante la partita non viene usato
// nessun interprete Python.
//
// Firma attesa del forward TorchScript:
//
//     forward(x, edge_index, edge_attr, u, batch) -> Tensor [B, 1]
//
// dove i tensori hanno lo stesso layout prodotto da BoardEncoder:
//
//     x          float32 [sumN, GNNNodeDim]
//     edge_index int64   [2,    sumE]
//     edge_attr  float32 [sumE, GNNEdgeDim]
//     u          float32 [B,    GNNGlobalDim]
//     batch      int64   [sumN]
//
// Il valore restituito e' assunto in [-1, 1] e dal punto di vista del giocatore
// che deve muovere, come descritto in docs/spiegazione_value_network.md.
class TorchScriptValueEvaluator
{
public:
    // device puo' essere "cpu", "cuda", "cuda:N" oppure "auto".
    // "auto" usa CUDA quando disponibile e altrimenti la CPU.
    explicit TorchScriptValueEvaluator(const std::string& modelPath,
                                       const std::string& device = "cpu");
    ~TorchScriptValueEvaluator();

    TorchScriptValueEvaluator(const TorchScriptValueEvaluator&) = delete;
    TorchScriptValueEvaluator& operator=(const TorchScriptValueEvaluator&) = delete;

    // Parametro opzionale di performance per l'MCTS.
    //
    // Se il motore usa piu' worker MCTS, spesso conviene tenere il parallelismo
    // intra-op di libtorch a 1 e parallelizzare a livello di ricerca.
    static void SetTorchThreads(int numThreads);

    // Attiva il coordinatore condiviso per il self-play parallelo. Ogni MCTS
    // invia una sola foglia e resta in attesa; il coordinatore aggrega richieste
    // provenienti da partite diverse fino a maxBatchSize o maxWaitMicros.
    void EnableCrossGameBatching(int maxBatchSize, int maxWaitMicros = 2000);

    // Metodo comodo: codifica una Board e la valuta come singolo grafo.
    float EvaluateBoard(const Board& board);

    // Valuta un grafo gia' codificato.
    //
    // Usa torch::from_blob, quindi i tensori puntano direttamente a graph.x,
    // graph.edge_index, graph.edge_attr e graph.u senza copia. Questi vettori
    // devono restare vivi fino alla fine del forward; qui la condizione e'
    // soddisfatta perche' graph e' un parametro della funzione.
    float EvaluateGraph(const GNNGraph& graph);

    // Metodo comodo: codifica piu' Board e le valuta con un solo forward.
    //
    // Questa e' la forma consigliata per l'MCTS. Le valutazioni delle foglie
    // sono piccole; il batch ammortizza il costo di libtorch e permette al
    // pooling stile PyG di usare il vettore batch per separare i grafi.
    std::vector<float> EvaluateBoards(const std::vector<Board>& boards);

    // Valuta un batch di grafi gia' codificati.
    //
    // PyTorch Geometric crea i batch concatenando tutti i nodi e tutti gli
    // archi, poi somma agli id degli archi l'offset dei nodi gia' inseriti.
    // Questa funzione replica manualmente la stessa operazione in C++.
    std::vector<float> EvaluateGraphs(const std::vector<GNNGraph>& graphs);

    // Restituisce una distribuzione di prior sulle mosse legali usando la
    // policy head TorchScript. Se il modello esportato non contiene la policy
    // head, restituisce un vettore vuoto e il chiamante puo' fare fallback.
    std::vector<float> EvaluateMovePriors(const Board& board, const std::vector<Move>& moves);

    const torch::Device& GetDevice() const { return m_device; }

private:
    struct ValueRequest;

    void ValueBatchWorker();

    static int64_t NodeCount(const GNNGraph& graph);// num nodi e numero edge per creare il tensore
    static int64_t EdgeCount(const GNNGraph& graph);
    static void ValidateGraph(const GNNGraph& graph); //Controlla che il grafo sia valido prima di mandarlo alla rete.
    static void ValidateOutput(const torch::Tensor& out, int64_t expectedBatchSize); 
/*Controlla che la rete abbia restituito il numero giusto di valori.
Se valuti una board, la rete deve restituire 1 valore.
Se valuti 8 board in batch, deve restituire 8 valori.*/

    torch::Device m_device;
    torch::jit::script::Module m_module;
    std::mutex m_moduleMutex;

    std::mutex m_queueMutex;
    std::condition_variable m_queueReady;
    std::deque<std::shared_ptr<ValueRequest>> m_valueQueue;
    std::thread m_valueBatchThread;
    bool m_stopValueBatchThread = false;
    int m_crossGameBatchSize = 1;
    int m_crossGameMaxWaitMicros = 2000;
    bool m_hasPolicyHead = false;
};

} // namespace HiveGotThis

#endif // NEURALEVALUATOR_H
