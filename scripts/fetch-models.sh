#!/usr/bin/env bash
# Reason Ready — turnkey base-model fetch.
#
#   ./scripts/fetch-models.sh              # embedder + reranker (the defaults)
#   ./scripts/fetch-models.sh embedder     # just the Qwen3 embedder
#   ./scripts/fetch-models.sh reranker     # just the Qwen3 reranker
#   ./scripts/fetch-models.sh --check      # verify what's on disk, download nothing
#
# The candle backends load weights from a local directory (docs/MODELS.md); this
# script is what puts real weights there. The base models are ~1.2 GB of
# safetensors each — too big to vendor in git — so they are pulled on demand and
# verified byte-exact against the manifest below. It is idempotent and resumable:
# a file already present at its exact expected size is left untouched, and a
# partial download is resumed, so re-running after a dropped connection is safe.
#
# Environment knobs:
#   RRO_MODELS_DIR   where models land        (default: <repo>/models)
#   HF_ENDPOINT      Hugging Face base URL     (default: https://huggingface.co;
#                    set to e.g. https://hf-mirror.com behind a firewall)
#   HF_REV           git revision to pull      (default: main)
#   HF_TOKEN         bearer token for HF        (optional; for rate limits)
#
# Weights are apache-2.0 (Qwen/Qwen3-Embedding-0.6B, Qwen/Qwen3-Reranker-0.6B).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${RRO_MODELS_DIR:-$ROOT/models}"
HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
HF_REV="${HF_REV:-main}"

# ── the manifest ────────────────────────────────────────────────────────────
# name  repo  file  bytes
# Sizes captured 2026-07-16 from revision `main`; they are the integrity check —
# a truncated download or an HTML error page will not match the exact byte count.
# Only the files the candle loaders read (config.json + tokenizer.json +
# model.safetensors) plus the small sentence-transformers/pooling descriptors the
# model card ships. The huge one is model.safetensors (~1.19 GB each).
MANIFEST="$(cat <<'EOF'
qwen3-embedding-0.6b Qwen/Qwen3-Embedding-0.6B config.json 727
qwen3-embedding-0.6b Qwen/Qwen3-Embedding-0.6B config_sentence_transformers.json 215
qwen3-embedding-0.6b Qwen/Qwen3-Embedding-0.6B modules.json 349
qwen3-embedding-0.6b Qwen/Qwen3-Embedding-0.6B tokenizer_config.json 9706
qwen3-embedding-0.6b Qwen/Qwen3-Embedding-0.6B tokenizer.json 11423705
qwen3-embedding-0.6b Qwen/Qwen3-Embedding-0.6B model.safetensors 1191586416
qwen3-reranker-0.6b Qwen/Qwen3-Reranker-0.6B config.json 727
qwen3-reranker-0.6b Qwen/Qwen3-Reranker-0.6B config_sentence_transformers.json 325
qwen3-reranker-0.6b Qwen/Qwen3-Reranker-0.6B tokenizer_config.json 9706
qwen3-reranker-0.6b Qwen/Qwen3-Reranker-0.6B tokenizer.json 11422654
qwen3-reranker-0.6b Qwen/Qwen3-Reranker-0.6B model.safetensors 1191588280
EOF
)"

# Which model dir belongs to which selection word.
EMBEDDER_DIR="qwen3-embedding-0.6b"
RERANKER_DIR="qwen3-reranker-0.6b"

# ── args ────────────────────────────────────────────────────────────────────
WANT=""            # empty = both
CHECK_ONLY=0
for arg in "$@"; do
  case "$arg" in
    embedder|reranker) WANT="$arg" ;;
    all|both)          WANT="" ;;
    --check)           CHECK_ONLY=1 ;;
    -h|--help)
      sed -n '2,26p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "unknown argument: $arg (expected: embedder | reranker | all | --check)" >&2; exit 2 ;;
  esac
done

selected_dir() {
  case "$WANT" in
    embedder) echo "$EMBEDDER_DIR" ;;
    reranker) echo "$RERANKER_DIR" ;;
    *)        echo "$EMBEDDER_DIR $RERANKER_DIR" ;;
  esac
}

# ── helpers ─────────────────────────────────────────────────────────────────
filesize() {
  # GNU stat, then BSD/macOS stat, then wc as a last resort.
  stat -c%s "$1" 2>/dev/null || stat -f%z "$1" 2>/dev/null || wc -c <"$1" 2>/dev/null || echo -1
}

human() {
  awk -v b="$1" 'BEGIN{ split("B KB MB GB TB",u); i=1; while(b>=1024 && i<5){b/=1024;i++}
    printf (i==1?"%d %s":"%.1f %s"), b, u[i] }'
}

have_cli() {
  command -v hf >/dev/null 2>&1 || command -v huggingface-cli >/dev/null 2>&1
}

hf_cli() {
  if command -v hf >/dev/null 2>&1; then hf "$@"; else huggingface-cli "$@"; fi
}

# Download one file, resuming if a partial exists. Verifies size after.
download_one() {
  local repo="$1" file="$2" want="$3" dest="$4"
  local url="$HF_ENDPOINT/$repo/resolve/$HF_REV/$file?download=true"

  if have_cli && [[ "$HF_ENDPOINT" == "https://huggingface.co" ]]; then
    # The CLI handles LFS pointers, resume and auth cleanly. It writes into
    # --local-dir at the file's repo-relative path (these files are flat).
    hf_cli download "$repo" "$file" --revision "$HF_REV" \
      --local-dir "$(dirname "$dest")" >/dev/null
  elif command -v curl >/dev/null 2>&1; then
    curl -fL --retry 5 --retry-delay 2 --retry-connrefused -C - \
      ${HF_TOKEN:+-H "Authorization: Bearer $HF_TOKEN"} \
      -o "$dest" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -c ${HF_TOKEN:+--header="Authorization: Bearer $HF_TOKEN"} \
      -O "$dest" "$url"
  else
    echo "ERROR: need one of hf / huggingface-cli / curl / wget to download." >&2
    return 3
  fi

  local got; got="$(filesize "$dest")"
  if [[ "$got" != "$want" ]]; then
    echo "  ✗ $file: got $got bytes, expected $want — download incomplete or wrong revision" >&2
    return 4
  fi
}

# ── run ─────────────────────────────────────────────────────────────────────
echo "── Reason Ready: base model fetch ─────────────────────────────"
echo "target:   $MODELS_DIR"
echo "endpoint: $HF_ENDPOINT   rev: $HF_REV"
if have_cli; then echo "method:   huggingface CLI (resumable, LFS-aware)"; else echo "method:   $(command -v curl >/dev/null 2>&1 && echo curl || echo wget) (resumable)"; fi
echo

want_dirs="$(selected_dir)"
missing=0 fetched=0 verified=0

for dir in $want_dirs; do
  target="$MODELS_DIR/$dir"
  mkdir -p "$target"
  echo "▸ $dir"
  # Iterate the manifest rows for this dir.
  while read -r m_dir m_repo m_file m_bytes; do
    [[ -z "${m_dir:-}" || "$m_dir" != "$dir" ]] && continue
    dest="$target/$m_file"
    if [[ -f "$dest" && "$(filesize "$dest")" == "$m_bytes" ]]; then
      echo "  ✓ $m_file ($(human "$m_bytes")) — present"
      verified=$((verified+1))
      continue
    fi
    if [[ "$CHECK_ONLY" == "1" ]]; then
      echo "  … $m_file ($(human "$m_bytes")) — MISSING"
      missing=$((missing+1))
      continue
    fi
    echo "  ↓ $m_file ($(human "$m_bytes"))"
    download_one "$m_repo" "$m_file" "$m_bytes" "$dest"
    echo "  ✓ $m_file — verified"
    fetched=$((fetched+1))
  done <<< "$MANIFEST"
  echo
done

if [[ "$CHECK_ONLY" == "1" ]]; then
  if [[ "$missing" -gt 0 ]]; then
    echo "$missing file(s) missing. Run without --check to download."
    exit 1
  fi
  echo "All selected model files present and byte-exact."
  exit 0
fi

echo "── done: $fetched downloaded, $verified already present ───────"
echo
echo "Point the engine at them (candle backends):"
if [[ "$WANT" != "reranker" ]]; then
  echo "  export RRO_EMBEDDER=candle-qwen"
  echo "  export RRO_EMBEDDER_WEIGHTS=$MODELS_DIR/$EMBEDDER_DIR"
fi
if [[ "$WANT" != "embedder" ]]; then
  echo "  export RRO_RERANKER=candle-cross-encoder"
  echo "  export RRO_RERANKER_WEIGHTS=$MODELS_DIR/$RERANKER_DIR"
fi
echo "  cargo build --release --features candle --bin rro"
echo "Or one command:  RRO_EMBEDDER=candle-qwen ./scripts/quickstart.sh"
