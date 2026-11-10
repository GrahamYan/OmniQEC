#!/usr/bin/env bash
# ONE shared CLIProxyAPI on the login node, serving every compute-node job.
#
#   bash omniqec_agentic/hpc/nscc_cliproxy_shared.sh start
#   bash omniqec_agentic/hpc/nscc_cliproxy_shared.sh status
#   bash omniqec_agentic/hpc/nscc_cliproxy_shared.sh stop
#
# WHY. A Claude subscription grant is ONE OAuth credential. Every CLIProxyAPI
# instance refreshes it on its own schedule, and the provider's refresh-token
# reuse detection treats concurrent rotation as theft: the WHOLE grant gets
# revoked ("OAuth access token has been revoked"). Measured 2026-08-26 with 5
# per-job instances: b60 died three chain links in a row while the other four
# points coasted on unexpired access tokens. One instance = one refresher =
# no reuse to detect. This is what makes 5-way parallelism safe.
#
# The endpoint is published in ~/.cli-proxy-api/shared_endpoint.env; the grid
# job script uses it AUTOMATICALLY whenever that file exists (that default is
# deliberate: forgetting a flag must not silently start a second refresher).
# 'stop' removes the file, returning jobs to per-job instances (serial only!).
#
# The listener binds the cluster-internal interface, gated by the same inbound
# api-key the per-job configs use. Do not expose that port outside the cluster.
set -uo pipefail

CLIPROXY_DIR="${CLIPROXY_DIR:-$HOME/cliproxyapi}"
CRED_DIR="$HOME/.cli-proxy-api"
ENDPOINT_FILE="$CRED_DIR/shared_endpoint.env"
SHARED_PORT="${SHARED_PORT:-18317}"
SHARED_CONFIG="$CRED_DIR/shared_config.yaml"
READY_TRIES="${READY_TRIES:-30}"
LOG="${LOG:-$HOME/OmniQEC/runs/logs/cliproxy_shared.log}"

usage() { echo "usage: $0 start|status|stop" >&2; exit 2; }
[[ $# -eq 1 ]] || usage

# Owner account: the inbound key in the local config.yaml. Collaborator
# account (pure HTTP client, no cliproxyapi install): the CLIPROXY_KEY it
# copied into its own endpoint file. Without this fallback `status` on a
# collaborator account sent an empty bearer, got 401 and reported the shared
# instance as down even when it was up (2026-09-03).
api_key() {
  if [[ -f "$CLIPROXY_DIR/config.yaml" ]]; then
    grep -m1 -oE 'sk-[A-Za-z0-9]+' "$CLIPROXY_DIR/config.yaml"
  elif [[ -f "$ENDPOINT_FILE" ]]; then
    sed -n 's/^CLIPROXY_KEY=//p' "$ENDPOINT_FILE" | head -1
  fi
}

endpoint_host() {
  local host=""
  [[ -f "$ENDPOINT_FILE" ]] && host="$(sed -n 's/^CLIPROXY_HOST=//p' "$ENDPOINT_FILE" | head -1)"
  echo "${host:-127.0.0.1}"
}

is_up() {
  curl -sf --max-time 5 "http://$(endpoint_host):$SHARED_PORT/v1/models" \
    -H "Authorization: Bearer $(api_key)" -o /dev/null 2>/dev/null
}

case "$1" in
start)
  [[ -x "$CLIPROXY_DIR/cli-proxy-api" ]] || { echo "no CLIProxyAPI at $CLIPROXY_DIR" >&2; exit 4; }
  compgen -G "$CRED_DIR/*.json" > /dev/null || { echo "no credential in $CRED_DIR — run -claude-login first" >&2; exit 4; }
  if is_up; then
    echo "already running on port $SHARED_PORT — not starting a second refresher"
    exit 0
  fi
  # A stale process that is up but unhealthy would still hold the port.
  pkill -f "cli-proxy-api -config $SHARED_CONFIG" 2>/dev/null && sleep 2

  mkdir -p "$(dirname "$LOG")"
  sed -E -e "s/^port:.*/port: $SHARED_PORT/" -e 's/^host:.*/host: "0.0.0.0"/' \
    "$CLIPROXY_DIR/config.yaml" > "$SHARED_CONFIG"
  chmod 600 "$SHARED_CONFIG"

  # setsid detaches from this SSH session so a dropped connection does not take
  # the proxy down; fall back to bare nohup where setsid is missing (macOS).
  runner=(setsid nohup); command -v setsid > /dev/null || runner=(nohup)
  ( cd "$CLIPROXY_DIR" && umask 077 && exec "${runner[@]}" ./cli-proxy-api -config "$SHARED_CONFIG" ) \
    >> "$LOG" 2>&1 &

  ready=0
  for _ in $(seq 1 "$READY_TRIES"); do
    sleep 2
    if is_up; then ready=1; break; fi
  done
  (( ready )) || { echo "did not become ready on port $SHARED_PORT:" >&2; tail -20 "$LOG" >&2; exit 4; }

  {
    echo "CLIPROXY_HOST=$(hostname)"
    echo "CLIPROXY_PORT=$SHARED_PORT"
  } > "$ENDPOINT_FILE"
  chmod 600 "$ENDPOINT_FILE"
  echo "shared CLIProxyAPI ready: $(hostname):$SHARED_PORT"
  echo "endpoint published in $ENDPOINT_FILE — grid jobs will use it automatically"
  echo "NOTE: it dies if this login node reboots; re-run 'start' then (jobs retry, nothing is lost)"
  ;;
status)
  if is_up; then
    echo "up on port $SHARED_PORT"
    # host/port only: the inbound key must never land in a terminal or a paste
    [[ -f "$ENDPOINT_FILE" ]] && grep -E "^CLIPROXY_(HOST|PORT)=" "$ENDPOINT_FILE"
    pgrep -af "cli-proxy-api -config $SHARED_CONFIG" | head -2
  else
    echo "not running (or not healthy) at $(endpoint_host):$SHARED_PORT"
    [[ -f "$ENDPOINT_FILE" ]] && echo "WARNING: $ENDPOINT_FILE still exists — jobs will try this endpoint and fail; owner account: run 'start' (or 'stop'); collaborator account: ask the credential owner to 'start' it on the host named in the file"
    exit 1
  fi
  ;;
stop)
  if pkill -f "cli-proxy-api -config $SHARED_CONFIG" 2>/dev/null; then
    # Wait for the exit: "stop && start" would otherwise race the dying process
    # for the port.
    for _ in $(seq 1 10); do
      pgrep -f "cli-proxy-api -config $SHARED_CONFIG" > /dev/null 2>&1 || break
      sleep 1
    done
    echo "stopped"
  else
    echo "was not running"
  fi
  rm -f "$ENDPOINT_FILE"
  echo "endpoint file removed — jobs fall back to per-job instances (SERIAL ONLY: concurrent per-job instances revoke the grant)"
  ;;
*) usage ;;
esac
