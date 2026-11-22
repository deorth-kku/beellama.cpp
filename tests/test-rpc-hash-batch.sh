#!/usr/bin/env bash
set -euo pipefail

server=$1
client=$2
port=$((42000 + $$ % 10000))
endpoint="127.0.0.1:${port}"
test_dir=$(mktemp -d)

cleanup() {
    kill "${pid:-}" 2>/dev/null || true
    rm -rf "$test_dir"
}
trap cleanup EXIT

wait_for_port() {
    local port=$1
    for _ in {1..600}; do
        if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
            exec 3>&-
            exec 3<&-
            return 0
        fi
        sleep 0.05
    done
    return 1
}

# the rpc-server derives its cache dir from $LLAMA_CACHE/rpc
export LLAMA_CACHE="$test_dir/llama-cache"
mkdir -p "$LLAMA_CACHE/rpc"

"$server" --device CPU --host 127.0.0.1 --port "$port" -c >"$test_dir/server.log" 2>&1 &
pid=$!
wait_for_port "$port"

"$client" "$endpoint" "$LLAMA_CACHE/rpc"
