# Fase 5: il ciclo a generazioni — piano di implementazione

## Contesto

La gen-0 (bootstrap da 31k partite umane BoardSpace) è misurata: 7-0 sulla rete casuale, 80% su Mzinga, 30% su nokamute (10 partite, 5s/mossa, seed 42). L'esperimento Kaggle sul dataset completo ha dimostrato che i dati umani sono esauriti per il value (l'informazione è limitata dal numero di partite): l'unica sorgente di informazione nuova è il self-play. Altri team del torneo riportano *peggioramenti* col self-play ingenuo → le salvaguardie non sono opzionali ma requisiti.

Decisioni già prese con l'utente:
- **Training manuale su GPU gratuita** (Colab/Kaggle, notebook esistenti); i 500€ di budget GPU restano in riserva (la generazione è CPU-bound locale, il training è piccolo).
- **Script per fase lanciati a mano** (stile `download_boardspace.sh`), niente orchestratore monolitico.
- **Temperatura sulle visite MCTS** al posto delle aperture uniformi in SelfPlay (π ∝ N^(1/τ), τ=1 nei primi plies, mai campionare via una vittoria provata).
- Salvaguardie: cancello `uhp_match` (promozione solo se batte la precedente), scala esterna fissa (Mzinga/nokamute), mix dati self-play + umani, budget MCTS adeguato, monitoraggio patte.

## Componenti da implementare

### 1. C++: esporre la distribuzione delle visite alla radice

`include/MCTS.h` + `src/MCTS.cpp`:

```cpp
struct RootMoveStat { Move move; int visits; int8_t provenResult; };
// overload, default assente → comportamento attuale invariato (Engine.cpp non si tocca)
static Move SearchIterations(const Board& rootBoard, int maxIterations,
                             std::vector<RootMoveStat>* rootStats);
```

Implementazione: dopo il loop di iterazioni, se `rootStats != nullptr` riempirlo da `root->children` (move, visitCount, provenResult). Nessun cambiamento all'algoritmo. Nota nel commento: questa distribuzione è anche il futuro target della policy head.

### 2. C++: temperatura in SelfPlay (`src/selfplay_main.cpp`)

L'argomento 6 della CLI cambia semantica: da `openingPlies` (mosse uniformi a caso tra le legali) a `tempPlies` (default 10) — mosse campionate sulla distribuzione delle visite. Logica per `ply < tempPlies`:

1. `SearchIterations(board, iterations, &stats)`;
2. se un figlio ha `provenResult == -1` (vittoria provata) → giocarlo, niente campionamento;
3. altrimenti campionare ∝ `visits` (τ=1) **escludendo** le sconfitte provate (`provenResult == +1`); se restano solo sconfitte → argmax visite;
4. il campionamento usa la `std::mt19937` già seedata per partita → riproducibilità invariata (stesso seed = stessa partita).

Per `ply >= tempPlies`: `SearchIterations` deterministico come oggi. Aggiornare il commento di testa del file (le mosse di temperatura sono *sensate*, non uniformi: etichette più pulite e partite comunque tutte diverse). La riga di stderr per partita (`Partita N terminata: ...`) resta: la usa lo script parallelo per il monitoraggio patte.

### 3. Script: generazione parallela — `scripts/selfplay/run_parallel.sh`

```
scripts/selfplay/run_parallel.sh <modello.pt> <output.jsonl> <partite> [iterazioni=400] [processi=nproc-1]
```

- Rifiuta un `<output.jsonl>` già esistente (stesso pattern anti-duplicazione di `download_boardspace.sh`).
- Divide le partite tra W processi `build/SelfPlay`; worker w riceve seed base distinto (es. `seed0 + w*1000000`) e `game_id = w*100000`; ogni worker scrive un file temporaneo `output.jsonl.w`, stderr → `output.jsonl.w.log`.
- `wait`, poi concatena i temporanei in `<output.jsonl>` e li cancella.
- Riepilogo finale dai log: conteggio WhiteWins/BlackWins/Draw/PlyCapReached e posizioni totali; **warning esplicito se (Draw+PlyCapReached) > 20%** (salvaguardia: diluvio di z=0).
- Commento di testa: perché si parallelizzano le PARTITE e mai l'MCTS singolo; una transposition table per processo (~24 MB l'una, va bene).

### 4. Script: mix dei dati — `scripts/selfplay/mix_dataset.py`

Piccolo (~50 righe, stile leggibile del progetto):

```
python3 scripts/selfplay/mix_dataset.py --selfplay data/selfplay_gen1.jsonl \
    --human data/boardspace_*.jsonl --output data/mix_gen1/ [--ratio 1.0] [--seed 42]
```

- Copia il file self-play in `--output` e vi scrive `umano_mix.jsonl` campionando posizioni dai file umani finché `posizioni_umane ≈ ratio × posizioni_selfplay` (campionamento per riga, deterministico col seed; per il value va bene, come dimostrato con `--sample`).
- Risultato: una cartella da zippare per Drive/Kaggle. Mix 50/50 di default: il sapere umano resta nell'impasto contro la dimenticanza.

### 5. Training (manuale, nessuna modifica al codice)

Notebook esistenti, con nell'ultima cella già documentato: `--init-weights checkpoint_gen0.pt --output checkpoint_gen1.pt`. Con ~200k posizioni di mix il Colab gratuito basta (niente `--sample`). Da fare: nessun file nuovo, solo istruzioni nel registro (v. §7).

### 6. Cancello e scala esterna (strumenti già esistenti, solo procedura)

- Export locale: `export_hive_value_gnn.py --weights checkpoint_gen1.pt --output gen1.pt`.
- **Cancello**: `uhp_match match ... --white-model gen1.pt --black-model gen0.pt --games 40 --time 1`. Promozione se score ≥ **55%**; 45-55% = inconcludente (raddoppiare le partite o generare più dati); < 45% = scartata (rivedere iterazioni/mix).
- **Scala esterna** per ogni generazione promossa: 10 partite @5s seed 42 vs Mzinga e vs nokamute (identiche alle misure gen-0). Il cancello interno può essere ingannato, la scala no.

### 7. Registro: `docs/generazioni.md`

Nuovo documento (stile impersonale, presente): la procedura del ciclo passo-passo (i comandi delle fasi 3-6) e la **tabella della curva di progresso**, una riga per generazione: dati usati (partite self-play, mix), training (epoca migliore, val MSE, segno-ok), cancello (score vs precedente), scala esterna (% Mzinga, % nokamute), decisione. Prima riga: gen-0 con i numeri già misurati.

### 8. Documentazione e memoria

- `docs/spiegazione_value_network.md`: righe nuove nella tabella file di §1 (`scripts/selfplay/*`, `docs/generazioni.md`); in §3 (SelfPlay) sostituire la descrizione delle aperture uniformi con la temperatura sulle visite; breve rimando a `generazioni.md` per la procedura del ciclo.
- Memoria: fase 5 in corso, parametri scelti.

## Parametri iniziali proposti per la gen-1

- **2000 partite**, 400 iterazioni/mossa, tempPlies 10, maxPlies 200, ~8 processi → ~1,5-2h locali, ~100k posizioni self-play + ~100k umane di mix.
- Razionale 400 iterazioni: più iterazioni = etichette migliori ma meno partite/ora; 400 è il default attuale e il primo giro serve a calibrare — se il cancello fallisce, la prima manopola da alzare è questa.

## Esplicitamente FUORI da questa fase

- Policy head (dopo che il ciclo gira; la temperatura ne prepara già l'infrastruttura dati).
- Salvataggio della distribuzione visite nel JSONL (richiede progettare l'encoding delle mosse: è il cantiere policy).
- **Fase 5b — server di inferenza con batching GPU** (idea dell'utente, DECISO: dopo il primo ciclo). Architettura: molte partite concorrenti; una variante di `NeuralEvaluator` lato C++ serializza il grafo su socket locale invece di chiamare libtorch; un server **Python + torch_geometric** carica il checkpoint (niente export), raccoglie le richieste (parti a K richieste o dopo T ms) e fa un forward batched (`Batch.from_data_list`) su GPU. Punto chiave misurato: il 98% del tempo MCTS è inferenza → con l'inferenza remota un core CPU ospita decine di partite; batch ≈ partite concorrenti (l'MCTS tiene in volo 1 richiesta per partita). Stima: T4 batched ~50-100k eval/s vs ~6.300/s degli 8 core locali → 10-20× partite/giorno. Non conviene in locale (batch ~8: il roundtrip mangia il guadagno): si attiva SE il ciclo validato ha fame di partite, con GPU a noleggio (~0,20-0,40€/h sui 500€ di budget). Costi accettati: riproducibilità per-seed indebolita, debugging distribuito.
- Noleggio GPU/CPU cloud (500€ in riserva finché il collo di bottiglia non è misurato).

## File toccati

| Azione | File |
|---|---|
| modifica | `include/MCTS.h`, `src/MCTS.cpp` (overload SearchIterations + RootMoveStat) |
| modifica | `src/selfplay_main.cpp` (temperatura al posto delle aperture uniformi) |
| crea | `scripts/selfplay/run_parallel.sh`, `scripts/selfplay/mix_dataset.py` |
| crea | `docs/generazioni.md` |
| modifica | `docs/spiegazione_value_network.md` |

## Verifica

1. **Build**: ricompilare; `Engine.cpp` e `uhp_match` non toccati → `TestTournamentBenchmark` con `hive_value_gnn.pt` deve dare numeri in linea (nessuna regressione dall'overload).
2. **SelfPlay smoke**: 2 partite, 100 iterazioni, tempPlies 4, seed fisso. Attesi: (a) stesso seed → file identico (riproducibilità); (b) seed diversi → partite diverse (diversità); (c) ogni riga `json.loads`-abile con dimensioni feature coerenti; (d) z coerente con il `resultStr` di stderr.
3. **run_parallel.sh**: 8 partite su 2 processi → 8 partite nel file concatenato, game_id senza collisioni, riepilogo esiti corretto; rilancio sullo stesso output → rifiuto.
4. **mix_dataset.py**: proporzioni ~ratio, determinismo (due run → file identici).
5. **Mini-ciclo end-to-end su CPU** (solo plumbing, non qualità): 20 partite → mix piccolo → 2 epoche di training locale → export → `uhp_match --games 2 --depth 1` contro gen-0. Se il giro si chiude senza errori, la macchina è pronta per la gen-1 vera (che lancia l'utente).
