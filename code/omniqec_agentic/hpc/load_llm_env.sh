# Load every provider credential file, and say so when two of them collide.
#
#   source omniqec_agentic/hpc/load_llm_env.sh
#   load_llm_env                      # default ~/.config/omniqec
#
# Credentials live one file per provider so adding a provider is adding a
# file. The cost of that convenience is that two files can claim the same
# variable and the last one alphabetically wins in silence — which is exactly
# what happened on 2026-08-22: a retired CLIProxyAPI config (openai_proxy.env)
# sorted after openai.env and pointed every OpenAI call at a dead ngrok
# tunnel, while the file the operator had just written sat there unused.
#
# The override still happens (last wins, as any shell would), but it is now
# announced with the losing and winning files named.
load_llm_env() {
  local dir="${1:-$HOME/.config/omniqec}"
  local manifest env_file base var files winner dupes
  manifest="$(mktemp)"

  shopt -s nullglob
  for env_file in "$dir"/*.env; do
    base="$(basename "$env_file")"
    # '|| true': grep exits 1 when a file declares nothing in KEY=VALUE form
    # (an export-only file, a comment-only file, an empty one). Under the
    # `set -euo pipefail` every job script uses, that exit code aborted the
    # WHOLE launcher here -- no message, no 'loaded' line, credentials never
    # sourced, exit 1 from a function whose job is to be chatty about problems.
    grep -oE '^[A-Za-z_][A-Za-z0-9_]*=' "$env_file" 2>/dev/null \
      | tr -d '=' | sed "s|^|$base |" >> "$manifest" || true
    set -a
    # shellcheck disable=SC1090
    source "$env_file"
    set +a
    echo "loaded $base"
  done
  shopt -u nullglob

  dupes="$(awk '{print $2}' "$manifest" | sort | uniq -d)"
  if [[ -n "$dupes" ]]; then
    echo "WARNING: variables claimed by more than one env file (LAST file wins):" >&2
    while read -r var; do
      [[ -z "$var" ]] && continue
      files="$(awk -v v="$var" '$2 == v {print $1}' "$manifest" | tr '\n' ' ')"
      winner="$(awk -v v="$var" '$2 == v {print $1}' "$manifest" | tail -1)"
      echo "  $var  set by: ${files}-> $winner wins" >&2
    done <<< "$dupes"
    echo "  Retire the stale file (mv it to *.env.retired) if this is not what you want." >&2
  fi
  rm -f "$manifest"
}
