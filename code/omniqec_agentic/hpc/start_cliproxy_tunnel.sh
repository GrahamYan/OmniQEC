#!/bin/bash
# Mac side: expose a local CLIProxyAPI through a tunnel and point the Worker at it.
#
#   bash omniqec_agentic/hpc/start_cliproxy_tunnel.sh
#
# The tunnel hostname is ephemeral — it changes every restart, and a stale one
# is why the NSCC -> GPT route died between 2026-08-18 and 2026-08-22 (the
# cluster kept calling a hostname whose tunnel had gone offline, and the only
# symptom was ERR_NGROK_3200 buried in a 401-shaped failure). Capturing the new
# hostname and redeploying the Worker in the same breath is the whole point of
# this script: the two must never be allowed to drift.
#
# NSCC never talks to the tunnel directly (its firewall resets those hosts).
# Only the Worker does, and the Worker runs inside Cloudflare's network, so any
# publicly resolvable tunnel provider works.
set -euo pipefail

PORT="${CLIPROXY_PORT:-8317}"
WORKER_DIR="${WORKER_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/worker_deepseek" && pwd)}"
LOG="${TMPDIR:-/tmp}/cliproxy_tunnel.log"

command -v cloudflared >/dev/null 2>&1 || {
  echo "cloudflared not installed:  brew install cloudflared" >&2
  exit 2
}

# Fail before opening a tunnel to nothing: a tunnel to a dead port publishes a
# URL that 502s, which looks identical to a firewall problem from the cluster.
if ! curl -sf --max-time 5 "http://127.0.0.1:${PORT}/v1/models" -o /dev/null \
   && ! curl -s --max-time 5 "http://127.0.0.1:${PORT}/v1/models" -o /dev/null; then
  echo "nothing answering on 127.0.0.1:${PORT} — start CLIProxyAPI first" >&2
  exit 3
fi
echo "CLIProxyAPI is up on 127.0.0.1:${PORT}"

: > "$LOG"
cloudflared tunnel --url "http://127.0.0.1:${PORT}" --no-autoupdate > "$LOG" 2>&1 &
TUNNEL_PID=$!
echo "cloudflared pid=$TUNNEL_PID  log=$LOG"

URL=""
for _ in $(seq 1 40); do
  sleep 1
  URL="$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG" | head -1 || true)"
  [[ -n "$URL" ]] && break
done
if [[ -z "$URL" ]]; then
  echo "tunnel did not publish a hostname within 40s; see $LOG" >&2
  kill "$TUNNEL_PID" 2>/dev/null || true
  exit 4
fi
echo "tunnel URL: $URL"

( cd "$WORKER_DIR" && npx wrangler deploy --var "UPSTREAM_CLIPROXY:$URL" )

cat <<EOF

Worker now forwards /<token>/cliproxy/v1/* to $URL

On NSCC, point the provider at that path (keep the trailing /v1):
  OPENAI_BASE_URL=https://omniqec-deepseek.<subdomain>.workers.dev/<token>/cliproxy/v1

Leave this shell open: closing it drops the tunnel and every call starts
failing at the Worker. Re-run this script after any restart — the hostname
will be different and the Worker must be redeployed with the new one.
EOF

wait "$TUNNEL_PID"
