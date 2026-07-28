#!/usr/bin/env bash
#
# Scarica l'archivio partite di BoardSpace e lo converte in dataset JSONL,
# un anno per volta. Riavviabile senza danni: gli zip gia' scaricati vengono
# saltati, e un anno gia' convertito (il suo .jsonl esiste) non viene rifatto.
# Se una conversione viene interrotta a meta', cancellare il suo .jsonl
# prima di rilanciare (l'output e' in append: rilanciare senza cancellare
# duplicherebbe le partite).
#
# Uso:
#   scripts/boardspace/download_boardspace.sh [primo_anno] [ultimo_anno]   # default: 2006 2026
#
# Requisiti: build/HiveEngine compilato e un modello .pt qualsiasi per
# avviarlo (variabile HIVE_VALUE_NETWORK oppure hive_value_gnn.pt nella
# radice del progetto; per il replay la rete non viene mai interrogata).
#
# Output:
#   data/boardspace/<anno>/            gli zip originali (la "materia prima")
#   data/boardspace_<anno>.jsonl       il dataset convertito per il training
#   data/boardspace/<anno>/conversione.log   il dettaglio delle partite scartate

set -euo pipefail
cd "$(dirname "$0")/../.."   # lavora sempre dalla radice del progetto

BASE_URL="https://www.boardspace.net/hive/hivegames"
FIRST_YEAR="${1:-2006}"
LAST_YEAR="${2:-2026}"
MODEL="${HIVE_VALUE_NETWORK:-hive_value_gnn.pt}"

if [ ! -x build/HiveEngine ]; then
    echo "Errore: build/HiveEngine non trovato. Compila prima con ./build.sh" >&2
    exit 1
fi
if [ ! -f "$MODEL" ]; then
    echo "Errore: modello '$MODEL' non trovato. Generane uno con:" >&2
    echo "  python3 scripts/export_hive_value_gnn.py --output hive_value_gnn.pt" >&2
    exit 1
fi

for year in $(seq "$FIRST_YEAR" "$LAST_YEAR"); do
    dir="data/boardspace/$year"
    jsonl="data/boardspace_$year.jsonl"
    mkdir -p "$dir"

    echo "=== $year: download ==="
    # L'indice HTML della cartella archive-<anno>/ elenca gli zip periodici.
    # -f fa fallire curl sugli errori HTTP (senza, una pagina 404 verrebbe
    # salvata come se fosse uno zip); --retry assorbe i singhiozzi di rete.
    zips=$(curl -sf --retry 3 --retry-delay 2 "$BASE_URL/archive-$year/" \
           | grep -o 'href="games-[^"]*\.zip"' | sed 's/href="//;s/"//' || true)
    if [ -z "$zips" ]; then
        echo "    nessun archivio per il $year (cartella vuota o assente)"
        continue
    fi
    failed=0
    for z in $zips; do
        if [ ! -f "$dir/$z" ]; then
            if ! curl -sf --retry 3 --retry-delay 2 -o "$dir/$z.tmp" "$BASE_URL/archive-$year/$z"; then
                echo "    ATTENZIONE: download fallito per $z" >&2
                rm -f "$dir/$z.tmp"
                failed=1
                continue
            fi
            mv "$dir/$z.tmp" "$dir/$z"   # nome definitivo solo se completo
            sleep 0.4   # gentilezza verso il server
        fi
    done
    echo "    $(ls "$dir" | grep -c '\.zip$') zip in $dir"
    if [ "$failed" -ne 0 ]; then
        echo "    anno incompleto: conversione rimandata, rilancia lo script per riprovare" >&2
        continue
    fi

    if [ -f "$jsonl" ]; then
        echo "=== $year: gia' convertito ($jsonl esiste), salto ==="
        continue
    fi

    echo "=== $year: conversione ==="
    # game_id distinti per anno (anno*10000): non indispensabile (il training
    # distingue le partite per coppia file+id) ma comodo per il debugging.
    python3 scripts/boardspace/boardspace_to_jsonl.py "$dir" \
        --output "$jsonl" \
        --model "$MODEL" \
        --game-id-start $((year * 10000)) \
        2> "$dir/conversione.log"
done

echo
echo "=== Riepilogo finale ==="
for f in data/boardspace_*.jsonl; do
    [ -f "$f" ] && echo "  $f: $(wc -l < "$f") posizioni"
done
