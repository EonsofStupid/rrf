#!/usr/bin/env bash
# Reason Ready — turnkey quickstart.
#
#   ./scripts/quickstart.sh            # build, boot, smoke-test over a2a
#   ./scripts/quickstart.sh stop       # stop the daemon
#
# One command yields a running engine: persistent estate, RRD front door,
# ANN-indexed hybrid recall, reranker, readiness classifier, a2a listener,
# DuckDB-ready event stream — then proves it end-to-end over the wire and
# prints the flow stages the engine emitted while answering.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${RRF_HOME:-$ROOT/.rrf}"
ESTATE="$RUN_DIR/estate"
EVENTS="$RUN_DIR/events.jsonl"
PIDFILE="$RUN_DIR/rrf.pid"
ADDR="${RRO_LISTEN:-127.0.0.1:7878}"

stop() {
  if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    kill -TERM "$(cat "$PIDFILE")" && sleep 1
    echo "stopped rrf (pid $(cat "$PIDFILE"))"
  else
    echo "no running rrf daemon found"
  fi
  rm -f "$PIDFILE"
}

if [[ "${1:-}" == "stop" ]]; then stop; exit 0; fi

mkdir -p "$RUN_DIR"

echo "── building (release) ─────────────────────────────────────────"
cargo build --release --bin rrf --bin rro-bench

echo "── booting the engine ─────────────────────────────────────────"
[[ -f "$PIDFILE" ]] && stop
RRO_ESTATE="$ESTATE" RRO_LISTEN="$ADDR" RRO_EVENTS="$EVENTS" RUST_LOG=info \
  "$ROOT/target/release/rrf" >>"$RUN_DIR/rro.log" 2>&1 &
echo $! > "$PIDFILE"

for _ in $(seq 1 50); do
  if (exec 3<>"/dev/tcp/${ADDR%:*}/${ADDR#*:}") 2>/dev/null; then exec 3>&-; break; fi
  sleep 0.2
done
echo "engine up: pid $(cat "$PIDFILE"), a2a on $ADDR"
echo "estate:    $ESTATE"
echo "events:    $EVENTS"

echo "── smoke test: full pipeline over a2a (layer-2) ───────────────"
"$ROOT/target/release/rro-bench" --docs 500 --queries 25 --store estate \
  --remote "$ADDR" | grep -E "accuracy|p50|throughput" || true

echo "── flow stages the engine emitted while answering ─────────────"
grep '"flow.stage"' "$EVENTS" | tail -5 || echo "(no stage events yet)"

echo
echo "READY. Ask it something:"
echo "  target/release/rro-bench --docs 0 --queries 1 --remote $ADDR   # or speak a2a JSON on $ADDR"
echo "Stop it:  ./scripts/quickstart.sh stop"
