# Registro delle generazioni e degli esperimenti

Questo documento tiene la storia misurabile del progetto: quali modelli sono
stati allenati, con quali dati e iperparametri, e come sono andati sul campo.
Le condizioni di misura sono fisse e non si cambiano mai, così i numeri di
mesi diversi restano confrontabili:

- **Cancello interno**: partite `uhp_match` contro il campione in carica,
  a coppie con stessa apertura e colori invertiti, 1s/mossa. Promozione con
  score ≥ 55%; 45-55% = zona grigia, si raddoppiano le partite e decide il
  quadro complessivo (scala esterna inclusa).
- **Scala esterna**: 10 partite a 5s/mossa, seed 42, contro nokamute
  (bersaglio del ciclo) e Mzinga (canarino delle regressioni). Dalla
  campagna di agosto 2026 Mzinga si misura solo alle promozioni.

## La progressione su nokamute (la curva che conta)

| Modello | Score vs nokamute | Note |
|---|---|---|
| gen-0 (solo value) | 30% (2-6-2) | bootstrap da 31k partite umane |
| policy v1 (move features) | 45% (4-5-1) | +15 punti dai prior nel PUCT |
| **policy v2 (edge prediction)** | **55% (9-7-4 su 20)** | +10 punti dalla policy sugli embedding |

## Campione in carica

**`weights/hive_policy_v2.pt`** (checkpoint `checkpoint_f2_v2.pt`, agosto
2026): tronco GATv2 hidden 64, value head + policy v2 "edge prediction"
sugli embedding dei nodi. Training da zero sul dataset umano completo
(1,58M posizioni) con augmentation D6 e cosine decay, 18 epoche, best
all'epoca 17: val MSE 0.681, policy CE 2.5645, segno-ok 72,1%.

## Cronologia delle promozioni

| Data | Modello | Cancello | nokamute | Mzinga | Decisione |
|---|---|---|---|---|---|
| lug 2026 | gen-0 value-only (`hive_value_gnn_gen0_colab.pt`) | — (primo) | 30% | 80% | riferimento iniziale |
| 29 lug 2026 | policy v1 (`hive_policy_gen0.pt`) | 61% vs gen-0 (21-12-7/40) | 45% | 90% | promosso |
| 1 ago 2026 | v1 + dataset intero + augment (`hive_policy_gen0_aug.pt`) | 65% vs v1 (6-3-1/10) | 45% | 80% | promosso (alla pari fuori, meglio dentro) |
| 4 ago 2026 | **policy v2 (`hive_policy_v2.pt`)** | 54,2% vs aug (29-24-7/60) | **55%** (9-7-4/20) | **100%** (10-0) | **promosso**: cancello al limite ma ogni asse esterno a favore, e CE −0,25 |

## Campagna di esperimenti (3-4 agosto 2026, ~26h GPU Kaggle)

| Esperimento | Ipotesi | Metriche (val) | Campo | Esito |
|---|---|---|---|---|
| F1: hidden 32 | −50% larghezza ⇒ +40% iterazioni MCTS compensano | MSE 0.6831 (≈pari), CE 2.8354 (+0,6%) | cancello 51% (40 partite), iterazioni fisse 57% (20), nokamute 45% (20), Mzinga 10-0 | **equivalente al campione a metà costo**: non promosso, ma candidato motore per il self-play (≈+40% partite/ora) |
| F2: policy v2 | gli embedding battono le 32 feature a mano | **CE 2.5645 vs 2.8172**, MSE pari | vedi promozione sopra | **promosso** |
| F3: v2 su tronco h32 (6 ago) | policy migliore + rete veloce insieme | MSE 0.6816, CE 2.6136 (+1,9% vs campione) | cancello 52% (13-12-5/30), nokamute 40% (7-11-2/20) vs 55% del campione | **non promosso**: pari nel derby, sotto sulla scala esterna — il tronco dimezzato costa alla testa v2 piu' che alla v1; resta il generatore piu' veloce con policy v2 per il self-play |
| S0: baseline screening | riferimento (6 epoche, 50% dati) | MSE 0.6873, CE 2.9070 | — | metro degli screening |
| S3: dropout 0.1 | con l'augmentation, 0.2 regolarizza troppo | MSE 0.6845, CE 2.8803: batte S0 su entrambe | — | **vince lo screening**: candidato al run pieno (combinato con la v2) nella prossima finestra |
| S1/S2: policy-weight 0.5/2.0 | bilanciamento delle due loss | non eseguiti (fine quota/fase) | — | riproposti per la prossima finestra |
| S5: 2 teste di attenzione | −17% latenza misurata | non eseguito (deprioritizzato dopo F1) | — | eventuale, dopo il self-play |

Lezioni di metodo fissate da questa campagna:

1. **Cancelli a 10 partite mentono**: F1 ha esordito con un 30% (2-6-2) che
   40 partite dopo era un 51-57%. Il protocollo minimo per decidere è 30
   partite; 10 bastano solo per scartare disastri.
2. **Le metriche di validation non bastano in nessuna direzione**: metriche
   pari possono nascondere forza pari (F1) e metriche migliori non
   garantiscono dominio nel testa-a-testa (F2, 54%) — decide il campo, e
   la scala esterna pesa più del derby interno.
3. **La P100 di Kaggle non è utilizzabile** (PyTorch dell'immagine senza
   supporto Pascal): i kernel via API vanno lanciati con
   `--accelerator NvidiaTeslaT4`.

## Prossimi passi

- Ciclo di self-play (fase 5) con `hive_policy_v2.pt` come motore; la
  variante h32 e' il candidato generatore se il collo e' il volume di partite.
- Distillazione da nokamute (registrazione partite UHP + conversione con
  `policytargets move`): dataset misto per la gen-1.
- Screening rimanenti (S1, S2, S5) e loss WDL a 3 classi per il value.
