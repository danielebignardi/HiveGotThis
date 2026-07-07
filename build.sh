#!/bin/bash
# Compila HiveEngine. Uso: ./build.sh
#
# Vedi docs/Come_Compilare_Ed_Eseguire.md per la guida completa.
set -e
cd "$(dirname "$0")"

LIBTORCH_DIR="$(pwd)/third_party/libtorch"

if [ ! -d "$LIBTORCH_DIR" ]; then
    echo "Errore: libtorch non trovata in $LIBTORCH_DIR"
    echo ""
    echo "Scaricala una volta sola (vedi docs/Come_Compilare_Ed_Eseguire.md, Passo 1):"
    echo "  mkdir -p third_party"
    echo "  curl -L -o libtorch.zip \"https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcpu.zip\""
    echo "  unzip libtorch.zip -d third_party"
    echo "  rm libtorch.zip"
    exit 1
fi

# Numero di core per compilare in parallelo: nproc su Linux, sysctl su macOS.
NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

cmake -B build -S . -DCMAKE_PREFIX_PATH="$LIBTORCH_DIR" > /dev/null
cmake --build build --target HiveEngine -j"$NPROC"

echo ""
echo "Fatto. Eseguibile pronto in: build/HiveEngine"
