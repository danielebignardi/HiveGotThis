#include "NeuralEvaluator.h"
#include "MoveEncoder.h"

#include <torch/cuda.h>
#include <torch/utils.h> // torch::set_num_threads (non incluso da torch/script.h)
#include <c10/core/InferenceMode.h>

#include <algorithm>
#include <chrono>
#include <future>
#include <stdexcept>
#include <utility>

namespace HiveGotThis
{

namespace
{
torch::Device ResolveDevice(const std::string& requested)
{
    if (requested == "auto")
        return torch::cuda::is_available() ? torch::Device(torch::kCUDA)
                                           : torch::Device(torch::kCPU);

    torch::Device device(requested);
    if (device.is_cuda() && !torch::cuda::is_available())
        throw std::runtime_error("CUDA requested but no CUDA device is available");
    return device;
}
}

struct TorchScriptValueEvaluator::ValueRequest
{
    explicit ValueRequest(GNNGraph encodedGraph)
        : graph(std::move(encodedGraph))
    {
    }

    GNNGraph graph;
    std::promise<float> result;
};

struct TorchScriptValueEvaluator::PolicyRequest
{
    PolicyRequest(GNNGraph encodedGraph, std::vector<float> encodedMoves)
        : graph(std::move(encodedGraph))
        , moveFeatures(std::move(encodedMoves))
    {
    }

    GNNGraph graph;
    std::vector<float> moveFeatures;
    std::promise<std::vector<float>> result;
};

TorchScriptValueEvaluator::TorchScriptValueEvaluator(const std::string& modelPath,
                                                     const std::string& device)
    : m_device(ResolveDevice(device))
    , m_module(torch::jit::load(modelPath, m_device))
{
    // Carichiamo il file .pt esportato da Python con torch.jit.script.
    //
    // Il file contiene sia l'architettura compilata della rete sia i pesi.
    // Da questo momento in poi il C++ puo' usare la rete senza avviare Python.

    // Fondamentale in inferenza: disattiva Dropout e altra logica da training.
    //
    // Se non chiamassimo eval(), la rete potrebbe produrre risultati diversi
    // a ogni chiamata, perche' Dropout resterebbe attivo.
    m_module.eval();
    m_module.to(m_device);
    m_hasPolicyHead = static_cast<bool>(m_module.find_method("forward_policy"));
}

TorchScriptValueEvaluator::~TorchScriptValueEvaluator()
{
    {
        std::lock_guard<std::mutex> lock(m_queueMutex);
        m_stopBatchThreads = true;
    }
    m_valueQueueReady.notify_all();
    m_policyQueueReady.notify_all();
    if (m_valueBatchThread.joinable())
        m_valueBatchThread.join();
    if (m_policyBatchThread.joinable())
        m_policyBatchThread.join();
}

void TorchScriptValueEvaluator::SetTorchThreads(int numThreads)
{
    // libtorch puo' usare piu' thread interni per eseguire le operazioni.
    // Per grafi molto piccoli, come quelli di Hive, troppi thread possono
    // costare piu' di quanto aiutino. Questa funzione permette di regolarli.
    if (numThreads < 1)
        throw std::invalid_argument("numThreads must be >= 1");

    torch::set_num_threads(numThreads);
}

void TorchScriptValueEvaluator::EnableCrossGameBatching(int maxBatchSize,
                                                        int maxWaitMicros)
{
    if (maxBatchSize < 1)
        throw std::invalid_argument("maxBatchSize must be >= 1");
    if (maxWaitMicros < 1)
        throw std::invalid_argument("maxWaitMicros must be >= 1");
    if (m_valueBatchThread.joinable())
        throw std::logic_error("cross-game batching is already enabled");

    m_crossGameBatchSize = maxBatchSize;
    m_crossGameMaxWaitMicros = maxWaitMicros;
    m_stopBatchThreads = false;

    if (maxBatchSize > 1)
    {
        m_valueBatchThread = std::thread(&TorchScriptValueEvaluator::ValueBatchWorker, this);
        if (m_hasPolicyHead)
            m_policyBatchThread = std::thread(&TorchScriptValueEvaluator::PolicyBatchWorker, this);
    }
}

void TorchScriptValueEvaluator::ValueBatchWorker()
{
    while (true)
    {
        std::vector<std::shared_ptr<ValueRequest>> requests;

        {
            std::unique_lock<std::mutex> lock(m_queueMutex);
            m_valueQueueReady.wait(lock, [this]
            {
                return m_stopBatchThreads || !m_valueQueue.empty();
            });

            if (m_stopBatchThreads && m_valueQueue.empty())
                return;

            const auto deadline = std::chrono::steady_clock::now()
                + std::chrono::microseconds(m_crossGameMaxWaitMicros);
            m_valueQueueReady.wait_until(lock, deadline, [this]
            {
                return m_stopBatchThreads
                    || static_cast<int>(m_valueQueue.size()) >= m_crossGameBatchSize;
            });

            const size_t count = std::min(
                m_valueQueue.size(), static_cast<size_t>(m_crossGameBatchSize));
            requests.reserve(count);
            for (size_t i = 0; i < count; ++i)
            {
                requests.push_back(std::move(m_valueQueue.front()));
                m_valueQueue.pop_front();
            }
        }

        try
        {
            std::vector<GNNGraph> graphs;
            graphs.reserve(requests.size());
            for (const auto& request : requests)
                graphs.push_back(std::move(request->graph));

            std::vector<float> values = EvaluateGraphs(graphs);
            if (values.size() != requests.size())
                throw std::runtime_error(
                    "cross-game value batch returned an invalid result count");

            for (size_t i = 0; i < requests.size(); ++i)
                requests[i]->result.set_value(values[i]);
        }
        catch (...)
        {
            std::exception_ptr error = std::current_exception();
            for (const auto& request : requests)
                request->result.set_exception(error);
        }
    }
}

void TorchScriptValueEvaluator::PolicyBatchWorker()
{
    while (true)
    {
        std::vector<std::shared_ptr<PolicyRequest>> requests;

        {
            std::unique_lock<std::mutex> lock(m_queueMutex);
            m_policyQueueReady.wait(lock, [this]
            {
                return m_stopBatchThreads || !m_policyQueue.empty();
            });

            if (m_stopBatchThreads && m_policyQueue.empty())
                return;

            const auto deadline = std::chrono::steady_clock::now()
                + std::chrono::microseconds(m_crossGameMaxWaitMicros);
            m_policyQueueReady.wait_until(lock, deadline, [this]
            {
                return m_stopBatchThreads
                    || static_cast<int>(m_policyQueue.size()) >= m_crossGameBatchSize;
            });

            const size_t count = std::min(
                m_policyQueue.size(), static_cast<size_t>(m_crossGameBatchSize));
            requests.reserve(count);
            for (size_t i = 0; i < count; ++i)
            {
                requests.push_back(std::move(m_policyQueue.front()));
                m_policyQueue.pop_front();
            }
        }

        try
        {
            std::vector<std::vector<float>> priors = EvaluatePolicyBatch(requests);
            if (priors.size() != requests.size())
                throw std::runtime_error(
                    "cross-game policy batch returned an invalid result count");

            for (size_t i = 0; i < requests.size(); ++i)
                requests[i]->result.set_value(std::move(priors[i]));
        }
        catch (...)
        {
            std::exception_ptr error = std::current_exception();
            for (const auto& request : requests)
                request->result.set_exception(error);
        }
    }
}

float TorchScriptValueEvaluator::EvaluateBoard(const Board& board)
{
    // Questo e' il metodo piu' comodo da usare dall'esterno.
    //
    // Chi lo chiama passa una Board normale del motore.
    // Noi la convertiamo nel formato numerico della GNN tramite BoardEncoder.
    GNNGraph graph = BoardEncoder::encode(board);
    if (!m_valueBatchThread.joinable())
        return EvaluateGraph(graph);

    auto request = std::make_shared<ValueRequest>(std::move(graph));
    std::future<float> result = request->result.get_future();
    {
        std::lock_guard<std::mutex> lock(m_queueMutex);
        m_valueQueue.push_back(request);
    }
    m_valueQueueReady.notify_one();
    return result.get();
}

float TorchScriptValueEvaluator::EvaluateGraph(const GNNGraph& graph)
{
    // Prima di parlare con libtorch controlliamo che il grafo sia coerente.
    // Se qualcosa e' sbagliato, e' meglio fallire qui con un messaggio chiaro
    // invece di ricevere un errore criptico dal modello.
    ValidateGraph(graph);

    // Disattiva il calcolo dei gradienti.
    //
    // In C++ stiamo solo facendo inferenza, non training, quindi autograd non
    // serve. Questo riduce memoria usata e overhead.
    c10::InferenceMode inferenceMode;

    // Il BoardEncoder salva i dati in vettori piatti.
    // Qui ricostruiamo le dimensioni logiche:
    //
    // x          -> [nodeCount, 18]
    // edge_index -> [2, edgeCount]
    // edge_attr  -> [edgeCount, 9]
    // u          -> [1, 21]
    const int64_t nodeCount = NodeCount(graph);
    const int64_t edgeCount = EdgeCount(graph);

    // Opzioni di tipo per i tensori.
    //
    // x, edge_attr e u sono float32.
    // edge_index e batch devono essere int64, come richiesto da PyTorch/PyG.
    auto fopt = torch::TensorOptions().dtype(torch::kFloat32);
    auto iopt = torch::TensorOptions().dtype(torch::kInt64);

    // from_blob NON copia i dati.
    //
    // Il tensore x punta direttamente alla memoria di graph.x.
    // Per questo graph deve restare vivo fino alla fine di forward().
    torch::Tensor x = torch::from_blob(
        const_cast<float*>(graph.x.data()),
        {nodeCount, GNNNodeDim},
        fopt
    );

    // edge_index contiene gli archi in formato COO:
    //
    // riga 0: nodi sorgente
    // riga 1: nodi destinazione
    //
    // Nel vettore C++ e' salvato come:
    // [src_0, src_1, ..., src_E-1, dst_0, dst_1, ..., dst_E-1]
    torch::Tensor edgeIndex = torch::from_blob(
        const_cast<int64_t*>(graph.edge_index.data()),
        {2, edgeCount},
        iopt
    );

    // edge_attr ha una riga per ogni arco.
    // Ogni riga e' un one-hot di 9 elementi:
    // 6 direzioni planari, up, down, self.
    torch::Tensor edgeAttr = torch::from_blob(
        const_cast<float*>(graph.edge_attr.data()),
        {edgeCount, GNNEdgeDim},
        fopt
    );

    // u contiene le feature globali della posizione.
    // Anche se valutiamo un solo grafo, la rete vuole la dimensione batch:
    // [1, 21], non solo [21].
    torch::Tensor u = torch::from_blob(
        const_cast<float*>(graph.u.data()),
        {1, GNNGlobalDim},
        fopt
    );

    // batch dice a PyG a quale grafo appartiene ogni nodo.
    //
    // Qui abbiamo un solo grafo, quindi tutti i nodi appartengono al grafo 0.
    // Esempio con 4 nodi:
    // batch = [0, 0, 0, 0]
    torch::Tensor batch = torch::zeros({nodeCount}, iopt);

    // La firma del forward TorchScript e':
    //
    // forward(x, edge_index, edge_attr, u, batch)
    //
    // L'ordine qui deve combaciare esattamente con quello del modello Python.
    std::vector<torch::jit::IValue> inputs = {
        x.to(m_device),
        edgeIndex.to(m_device),
        edgeAttr.to(m_device),
        u.to(m_device),
        batch.to(m_device)
    };

    // Eseguiamo il forward della rete.
    //
    // out dovrebbe avere forma [1, 1] per un singolo grafo.
    torch::Tensor out;
    {
        std::lock_guard<std::mutex> lock(m_moduleMutex);
        out = m_module.forward(inputs).toTensor().to(torch::kCPU).to(torch::kFloat32);
    }
    ValidateOutput(out, 1);

    // Convertiamo il tensore di output in un float C++.
    //
    // Il valore e' interpretato come:
    // +1 posizione molto buona per il giocatore che deve muovere
    //  0 posizione equilibrata
    // -1 posizione molto cattiva per il giocatore che deve muovere
    return out.reshape({-1})[0].item<float>();
}

std::vector<float> TorchScriptValueEvaluator::EvaluateBoards(const std::vector<Board>& boards)
{
    // Versione comoda per piu' Board.
    //
    // Convertiamo ogni Board nel suo GNNGraph e poi usiamo EvaluateGraphs,
    // che fa un solo forward batchato.
    std::vector<GNNGraph> graphs;
    graphs.reserve(boards.size());

    for (const Board& board : boards)
        graphs.push_back(BoardEncoder::encode(board));

    return EvaluateGraphs(graphs);
}

std::vector<float> TorchScriptValueEvaluator::EvaluateGraphs(const std::vector<GNNGraph>& graphs)
{
    // Nessun grafo da valutare: restituiamo semplicemente un vettore vuoto.
    if (graphs.empty())
        return {};

    // Controlliamo ogni grafo prima di costruire il batch.
    for (const GNNGraph& graph : graphs)
        ValidateGraph(graph);

    // Anche qui siamo solo in inferenza: niente gradienti.
    c10::InferenceMode inferenceMode;

    // Questi vettori conterranno il batch concatenato.
    //
    // xb        = tutte le x una dopo l'altra
    // edgeAttrB = tutte le edge_attr una dopo l'altra
    // ub        = tutte le u una dopo l'altra
    // src/dst   = sorgenti e destinazioni degli archi, con offset corretto
    // batchIds  = per ogni nodo, id del grafo a cui appartiene
    std::vector<float> xb;
    std::vector<float> edgeAttrB;
    std::vector<float> ub;
    std::vector<int64_t> src;
    std::vector<int64_t> dst;
    std::vector<int64_t> batchIds;

    int64_t totalNodes = 0;
    int64_t totalEdges = 0;

    // Prima passata: calcoliamo quanti nodi e archi totali avra' il batch.
    // Questo ci permette di fare reserve() e ridurre riallocazioni.
    for (const GNNGraph& graph : graphs)
    {
        totalNodes += NodeCount(graph);
        totalEdges += EdgeCount(graph);
    }

    xb.reserve(static_cast<size_t>(totalNodes) * GNNNodeDim);
    edgeAttrB.reserve(static_cast<size_t>(totalEdges) * GNNEdgeDim);
    ub.reserve(graphs.size() * GNNGlobalDim);
    src.reserve(static_cast<size_t>(totalEdges));
    dst.reserve(static_cast<size_t>(totalEdges));
    batchIds.reserve(static_cast<size_t>(totalNodes));

    int64_t nodeOffset = 0;

    // Seconda passata: costruiamo il batch vero e proprio.
    for (size_t graphId = 0; graphId < graphs.size(); ++graphId)
    {
        const GNNGraph& graph = graphs[graphId];
        const int64_t nodeCount = NodeCount(graph);
        const int64_t edgeCount = EdgeCount(graph);

        xb.insert(xb.end(), graph.x.begin(), graph.x.end());
        edgeAttrB.insert(edgeAttrB.end(), graph.edge_attr.begin(), graph.edge_attr.end());
        ub.insert(ub.end(), graph.u.begin(), graph.u.end());

        // Per ogni nodo di questo grafo salviamo il suo graphId.
        //
        // Esempio:
        // grafo 0 con 3 nodi, grafo 1 con 2 nodi:
        // batchIds = [0, 0, 0, 1, 1]
        batchIds.insert(
            batchIds.end(),
            static_cast<size_t>(nodeCount),
            static_cast<int64_t>(graphId)
        );

        // edge_index e' salvato come [tutte le sorgenti..., tutte le destinazioni...].
        // Gli id dei nodi di ogni grafo vanno traslati nel batch concatenato.
        //
        // Esempio:
        // - grafo 0 ha nodi 0,1,2
        // - grafo 1 internamente ha nodi 0,1
        // Nel batch, i nodi del grafo 1 diventano 3,4.
        // Quindi ogni suo edge_index deve ricevere +3.
        for (int64_t edge = 0; edge < edgeCount; ++edge)
            src.push_back(graph.edge_index[static_cast<size_t>(edge)] + nodeOffset);

        for (int64_t edge = 0; edge < edgeCount; ++edge)
            dst.push_back(graph.edge_index[static_cast<size_t>(edgeCount + edge)] + nodeOffset);

        nodeOffset += nodeCount;
    }

    // Ricostruiamo edge_index batchato nello stesso formato usato dal progetto:
    //
    // [tutte le sorgenti..., tutte le destinazioni...]
    std::vector<int64_t> edgeIndexB;
    edgeIndexB.reserve(src.size() + dst.size());
    edgeIndexB.insert(edgeIndexB.end(), src.begin(), src.end());
    edgeIndexB.insert(edgeIndexB.end(), dst.begin(), dst.end());

    auto fopt = torch::TensorOptions().dtype(torch::kFloat32);
    auto iopt = torch::TensorOptions().dtype(torch::kInt64);

    // Da qui in poi e' lo stesso principio della valutazione singola,
    // ma con dimensioni totali:
    //
    // x          -> [totalNodes, 18]
    // edge_index -> [2, totalEdges]
    // edge_attr  -> [totalEdges, 9]
    // u          -> [numero_grafi, 21]
    // batch      -> [totalNodes]
    torch::Tensor x = torch::from_blob(
        xb.data(),
        {totalNodes, GNNNodeDim},
        fopt
    );

    torch::Tensor edgeIndex = torch::from_blob(
        edgeIndexB.data(),
        {2, totalEdges},
        iopt
    );

    torch::Tensor edgeAttr = torch::from_blob(
        edgeAttrB.data(),
        {totalEdges, GNNEdgeDim},
        fopt
    );

    torch::Tensor u = torch::from_blob(
        ub.data(),
        {static_cast<int64_t>(graphs.size()), GNNGlobalDim},
        fopt
    );

    torch::Tensor batch = torch::from_blob(
        batchIds.data(),
        {totalNodes},
        iopt
    );

    // Forward batchato.
    //
    // Se graphs.size() == B, l'output atteso e' [B, 1].
    std::vector<torch::jit::IValue> inputs = {
        x.to(m_device),
        edgeIndex.to(m_device),
        edgeAttr.to(m_device),
        u.to(m_device),
        batch.to(m_device)
    };

    torch::Tensor out;
    {
        std::lock_guard<std::mutex> lock(m_moduleMutex);
        out = m_module.forward(inputs).toTensor().to(torch::kCPU).to(torch::kFloat32);
    }
    ValidateOutput(out, static_cast<int64_t>(graphs.size()));

    // Portiamo l'output da [B, 1] a [B], cosi' e' facile copiarlo
    // in un std::vector<float>.
    out = out.reshape({static_cast<int64_t>(graphs.size())});

    out = out.contiguous();
    const float* values = out.data_ptr<float>();
    return std::vector<float>(values, values + graphs.size());
}

std::vector<float> TorchScriptValueEvaluator::EvaluateMovePriors(const Board& board, const std::vector<Move>& moves)
{
    if (moves.empty() || !m_hasPolicyHead)
        return {};

    GNNGraph graph = BoardEncoder::encode(board);

    // La posizione iniziale non ha ancora nodi: la GNN non puo' creare un
    // embedding di board, quindi l'MCTS fara' fallback su prior euristiche.
    if (graph.x.empty())
        return {};

    ValidateGraph(graph);

    std::vector<float> moveFeatureValues(moves.size() * MoveFeatureDim);
    for (size_t i = 0; i < moves.size(); ++i)
    {
        EncodeMoveFeaturesInto(
            board, moves[i], moveFeatureValues.data() + i * MoveFeatureDim);
    }

    auto request = std::make_shared<PolicyRequest>(
        std::move(graph), std::move(moveFeatureValues));
    if (!m_policyBatchThread.joinable())
    {
        std::vector<std::shared_ptr<PolicyRequest>> requests = {request};
        return EvaluatePolicyBatch(requests).front();
    }

    std::future<std::vector<float>> result = request->result.get_future();
    {
        std::lock_guard<std::mutex> lock(m_queueMutex);
        m_policyQueue.push_back(request);
    }
    m_policyQueueReady.notify_one();
    return result.get();
}

std::vector<std::vector<float>> TorchScriptValueEvaluator::EvaluatePolicyBatch(
    const std::vector<std::shared_ptr<PolicyRequest>>& requests)
{
    if (requests.empty())
        return {};

    c10::InferenceMode inferenceMode;

    int64_t totalNodes = 0;
    int64_t totalEdges = 0;
    int64_t totalMoves = 0;
    for (const auto& request : requests)
    {
        ValidateGraph(request->graph);
        if (request->moveFeatures.empty()
            || request->moveFeatures.size() % MoveFeatureDim != 0)
        {
            throw std::invalid_argument(
                "policy move features have an invalid size");
        }
        totalNodes += NodeCount(request->graph);
        totalEdges += EdgeCount(request->graph);
        totalMoves += static_cast<int64_t>(request->moveFeatures.size() / MoveFeatureDim);
    }

    std::vector<float> xb;
    std::vector<float> edgeAttrB;
    std::vector<float> ub;
    std::vector<float> moveFeaturesB;
    std::vector<int64_t> src;
    std::vector<int64_t> dst;
    std::vector<int64_t> batchIds;
    std::vector<int64_t> moveBatchIds;

    xb.reserve(static_cast<size_t>(totalNodes) * GNNNodeDim);
    edgeAttrB.reserve(static_cast<size_t>(totalEdges) * GNNEdgeDim);
    ub.reserve(requests.size() * GNNGlobalDim);
    moveFeaturesB.reserve(static_cast<size_t>(totalMoves) * MoveFeatureDim);
    src.reserve(static_cast<size_t>(totalEdges));
    dst.reserve(static_cast<size_t>(totalEdges));
    batchIds.reserve(static_cast<size_t>(totalNodes));
    moveBatchIds.reserve(static_cast<size_t>(totalMoves));

    int64_t nodeOffset = 0;
    for (size_t graphId = 0; graphId < requests.size(); ++graphId)
    {
        const PolicyRequest& request = *requests[graphId];
        const GNNGraph& graph = request.graph;
        const int64_t nodeCount = NodeCount(graph);
        const int64_t edgeCount = EdgeCount(graph);
        const int64_t moveCount =
            static_cast<int64_t>(request.moveFeatures.size() / MoveFeatureDim);

        xb.insert(xb.end(), graph.x.begin(), graph.x.end());
        edgeAttrB.insert(edgeAttrB.end(), graph.edge_attr.begin(), graph.edge_attr.end());
        ub.insert(ub.end(), graph.u.begin(), graph.u.end());
        moveFeaturesB.insert(
            moveFeaturesB.end(), request.moveFeatures.begin(), request.moveFeatures.end());
        batchIds.insert(
            batchIds.end(), static_cast<size_t>(nodeCount), static_cast<int64_t>(graphId));
        moveBatchIds.insert(
            moveBatchIds.end(), static_cast<size_t>(moveCount), static_cast<int64_t>(graphId));

        for (int64_t edge = 0; edge < edgeCount; ++edge)
            src.push_back(graph.edge_index[static_cast<size_t>(edge)] + nodeOffset);
        for (int64_t edge = 0; edge < edgeCount; ++edge)
        {
            dst.push_back(
                graph.edge_index[static_cast<size_t>(edgeCount + edge)] + nodeOffset);
        }
        nodeOffset += nodeCount;
    }

    std::vector<int64_t> edgeIndexB;
    edgeIndexB.reserve(src.size() + dst.size());
    edgeIndexB.insert(edgeIndexB.end(), src.begin(), src.end());
    edgeIndexB.insert(edgeIndexB.end(), dst.begin(), dst.end());

    auto fopt = torch::TensorOptions().dtype(torch::kFloat32);
    auto iopt = torch::TensorOptions().dtype(torch::kInt64);
    torch::Tensor x = torch::from_blob(
        xb.data(), {totalNodes, GNNNodeDim}, fopt);
    torch::Tensor edgeIndex = torch::from_blob(
        edgeIndexB.data(), {2, totalEdges}, iopt);
    torch::Tensor edgeAttr = torch::from_blob(
        edgeAttrB.data(), {totalEdges, GNNEdgeDim}, fopt);
    torch::Tensor u = torch::from_blob(
        ub.data(), {static_cast<int64_t>(requests.size()), GNNGlobalDim}, fopt);
    torch::Tensor batch = torch::from_blob(
        batchIds.data(), {totalNodes}, iopt);
    torch::Tensor moveFeatures = torch::from_blob(
        moveFeaturesB.data(), {totalMoves, MoveFeatureDim}, fopt);
    torch::Tensor moveBatch = torch::from_blob(
        moveBatchIds.data(), {totalMoves}, iopt);

    std::vector<torch::jit::IValue> inputs = {
        x.to(m_device),
        edgeIndex.to(m_device),
        edgeAttr.to(m_device),
        u.to(m_device),
        batch.to(m_device),
        moveFeatures.to(m_device),
        moveBatch.to(m_device)
    };

    torch::Tensor priorsTensor;
    {
        std::lock_guard<std::mutex> lock(m_moduleMutex);
        auto policyMethod = m_module.find_method("forward_policy");
        if (!policyMethod)
            throw std::runtime_error("TorchScript policy head disappeared at runtime");
        torch::Tensor logits = (*policyMethod)(inputs).toTensor();
        if (!logits.defined() || logits.numel() != totalMoves)
            throw std::runtime_error(
                "TorchScript policy head returned an unexpected number of logits");

        logits = logits.reshape({totalMoves});
        std::vector<torch::Tensor> normalized;
        normalized.reserve(requests.size());
        int64_t moveOffset = 0;
        for (const auto& request : requests)
        {
            const int64_t moveCount =
                static_cast<int64_t>(request->moveFeatures.size() / MoveFeatureDim);
            normalized.push_back(
                torch::softmax(logits.narrow(0, moveOffset, moveCount), 0));
            moveOffset += moveCount;
        }
        priorsTensor = torch::cat(normalized)
            .to(torch::kCPU)
            .to(torch::kFloat32)
            .contiguous();
    }

    std::vector<std::vector<float>> allPriors;
    allPriors.reserve(requests.size());
    const float* priorData = priorsTensor.data_ptr<float>();
    int64_t moveOffset = 0;
    for (const auto& request : requests)
    {
        const int64_t moveCount =
            static_cast<int64_t>(request->moveFeatures.size() / MoveFeatureDim);
        allPriors.emplace_back(
            priorData + moveOffset, priorData + moveOffset + moveCount);
        moveOffset += moveCount;
    }

    return allPriors;
}
int64_t TorchScriptValueEvaluator::NodeCount(const GNNGraph& graph)
{
    // Ogni nodo ha GNNNodeDim feature.
    // graph.x e' piatto, quindi il numero di nodi e':
    // dimensione_totale_x / feature_per_nodo.
    return static_cast<int64_t>(graph.x.size() / GNNNodeDim);
}

int64_t TorchScriptValueEvaluator::EdgeCount(const GNNGraph& graph)
{
    // edge_index contiene sorgenti e destinazioni.
    // La sua lunghezza e' quindi 2 * numero_archi.
    return static_cast<int64_t>(graph.edge_index.size() / 2);
}

void TorchScriptValueEvaluator::ValidateGraph(const GNNGraph& graph)
{
    // La rete si aspetta almeno un nodo.
    // In Hive, dopo la prima mossa ci sara' almeno un pezzo sulla board.
    if (graph.x.empty())
        throw std::invalid_argument("GNNGraph has no nodes; the value network expects at least one piece node");

    // x deve poter essere vista come matrice [N, 18].
    if (graph.x.size() % GNNNodeDim != 0)
        throw std::invalid_argument("GNNGraph.x size is not divisible by GNNNodeDim");

    // edge_index deve poter essere visto come matrice [2, E].
    if (graph.edge_index.size() % 2 != 0)
        throw std::invalid_argument("GNNGraph.edge_index must contain sources followed by destinations");

    const int64_t nodeCount = NodeCount(graph);
    const int64_t edgeCount = EdgeCount(graph);

    // Ogni arco deve avere una riga edge_attr da 9 elementi.
    if (graph.edge_attr.size() != static_cast<size_t>(edgeCount) * GNNEdgeDim)
        throw std::invalid_argument("GNNGraph.edge_attr size does not match edge_index edge count");

    // Il vettore globale u ha dimensione fissa 21.
    if (graph.u.size() != GNNGlobalDim)
        throw std::invalid_argument("GNNGraph.u size does not match GNNGlobalDim");

    // Ogni id dentro edge_index deve riferirsi a un nodo realmente esistente.
    for (int64_t edgeId : graph.edge_index)
    {
        if (edgeId < 0 || edgeId >= nodeCount)
            throw std::out_of_range("GNNGraph.edge_index contains a node id outside [0, nodeCount)");
    }
}

void TorchScriptValueEvaluator::ValidateOutput(const torch::Tensor& out, int64_t expectedBatchSize)
{
    // Controlliamo che il modello abbia effettivamente restituito un tensore.
    if (!out.defined())
        throw std::runtime_error("TorchScript value network returned an undefined tensor");

    // Per B grafi vogliamo B valori.
    //
    // Forme accettabili:
    // [B, 1], [B] o qualunque forma con numel() == B.
    if (out.numel() != expectedBatchSize)
        throw std::runtime_error("TorchScript value network returned an unexpected number of values");
}

} // namespace HiveGotThis
