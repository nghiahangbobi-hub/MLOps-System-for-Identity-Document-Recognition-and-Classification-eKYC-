#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
session=ekyc-codex
roles=(orchestrator data model azure)
if [[ ${1:-} == run || ${1:-} == sandbox-check ]]; then
  role=${2:?role required}
  case "$role" in
    orchestrator) assignment='Coordinate the three terminal workers in tmux session ekyc-codex. Own integration, docs, reporting and verification. Read all checkpoints regularly, send tasks using tmux send-keys only to idle workers, and do available integration work while workers run. Continue after compaction. Do not modify worker-owned paths concurrently.' ;;
    data) assignment='Implement and verify MIDV-2020 data ETL and Label Studio integration in data_pipeline/ and labeling/. Open clips.tar read-only. Read the orchestrator checkpoint for the manifest contract.' ;;
    model) assignment='Implement and verify PyTorch MobileNetV3 Small training, MLflow and scoring under modeling/. Read the orchestrator checkpoint for data/API contracts. Write artifact interface in your checkpoint early.' ;;
    azure) assignment='Implement Azure training/deployment definitions, demo, telemetry and benchmark scripts under azure/ and demo/. Use the checkpoint contracts. Leave actual provisioning to the orchestrator.' ;;
    *) exit 2 ;;
  esac
  mkdir -p .agent-state
  checkpoint=".agent-state/$role.md"
  codex_bin=$(readlink -f "$(command -v codex)")
  flags=(--ask-for-approval never)
  if [[ $role == orchestrator ]]; then
    [[ $1 == run ]] || exit 2
    flags+=(--sandbox danger-full-access)
  else
    output="runs/midv-5gb/worker-$role"
    scratch="runs/workers/$role"
    checkpoint="$scratch/checkpoint.md"
    case "$role" in
      data) owned=(data_pipeline labeling); other=modeling ;;
      model) owned=(modeling); other=azure ;;
      azure) owned=(azure demo); other=data_pipeline ;;
    esac
    mkdir -p "$output" "$scratch"
    if [[ ! -e $checkpoint ]]; then
      if [[ -f .agent-state/$role.md ]]; then
        cp ".agent-state/$role.md" "$checkpoint"
      else
        touch "$checkpoint"
      fi
    fi
    filesystem=$(python3 - "$PWD" "$codex_bin" "$output" "$scratch" "${owned[@]}" <<'PY'
import json
from pathlib import Path
import sys

root, executable, *writable = sys.argv[1:]
rules = {":minimal": "read", executable: "read", root: "read"}
rules.update({str(Path(root) / path): "write" for path in writable})
print("{" + ",".join(f"{json.dumps(path)}={json.dumps(mode)}" for path, mode in rules.items()) + "}")
PY
)
    profile="ekyc_$role"
    flags+=(-c "permissions.$profile.filesystem=$filesystem"
            -c "permissions.$profile.network.enabled=false"
            -c "default_permissions=\"$profile\""
            -c 'web_search="disabled"'
            -c 'shell_environment_policy.inherit="none"'
            -c 'shell_environment_policy.set.PATH="/usr/local/bin:/usr/bin:/bin"'
            -c "shell_environment_policy.set.HOME=\"$PWD/$scratch\""
            -c "shell_environment_policy.set.TMPDIR=\"$PWD/$scratch\"")
    if [[ $1 == sandbox-check ]]; then
      exec env -u TMUX -u TMUX_PANE "$codex_bin" "${flags[@]}" sandbox -P "$profile" -- \
        bash -eu -c '
          test -r AGENTS.md
          : >> "$1"
          for directory in "$2" "$3" "${@:7}"; do
            marker=$(mktemp "$directory/.permission-check.XXXXXX")
            rm "$marker"
          done
          marker=".permission-check.$$"
          for directory in "$4" .agent-state ops /tmp; do
            if touch "$directory/$marker"; then
              rm "$directory/$marker"
              exit 1
            fi
          done
          ! cat "$5" >/dev/null || exit 1
          ! tmux -S "$6" list-sessions || exit 1
          ! curl -fsS --max-time 2 http://127.0.0.1:8080/health || exit 1
          printf "PASS: allowed writes, denied cross-role writes, credentials, tmux and HTTP\n"
        ' bash "$checkpoint" "$output" "$scratch" "$other" "$HOME/.codex/auth.json" \
        "${EKYC_TEST_TMUX_SOCKET:-/tmp/tmux-$(id -u)/default}" "${owned[@]}"
    fi
    assignment+=" Write your checkpoint and denied-operation requests in $checkpoint. Use $scratch for temporary files and $output for verification evidence. Treat .agent-state as read-only."
  fi
  printf '\nRole: %s | Checkpoint: %s\n\n' "$role" "$checkpoint"
  status=0
  if [[ $role == orchestrator ]]; then
    runner=("$codex_bin")
  else
    runner=(env -u TMUX -u TMUX_PANE TMPDIR="$PWD/$scratch" "$codex_bin")
  fi
  "${runner[@]}" "${flags[@]}" "Read AGENTS.md and the latest entry in .agent-state/orchestrator.md, plus .agent-state/$role.md. You are the $role. $assignment Take only the latest assigned task from the checkpoint or the orchestrator. If there is no open task, report ready. Write progress, blockers and next steps in $checkpoint." || status=$?
  printf '\nCodex exited (%s). Restart with: bash ops/codex-team.sh restart %s\n' "$status" "$role"
  exit "$status"
fi
if [[ ${1:-} == restart ]]; then
  role=${2:?role required}
  case "$role" in orchestrator|data|model|azure) ;; *) exit 2 ;; esac
  target="$session:$role"
  if [[ $(tmux display-message -p -t "$target" '#{pane_dead}') != 1 ]]; then
    printf 'Role %s already has a live pane.\n' "$role" >&2
    exit 1
  fi
  exec tmux respawn-pane -t "$target" "bash ops/codex-team.sh run $role"
fi
if ! tmux has-session -t "=$session" 2>/dev/null; then
  tmux new-session -d -s "$session" -n orchestrator -c "$PWD"
  tmux set-option -t "$session" remain-on-exit on
  for role in data model azure; do
    tmux new-window -d -t "$session" -n "$role" -c "$PWD"
  done
  for role in "${roles[@]}"; do
    tmux respawn-pane -k -t "$session:$role" "bash ops/codex-team.sh run $role"
  done
  tmux select-window -t "$session:orchestrator"
fi
if [[ ${1:-} == windows ]]; then
  for role in "${roles[@]}"; do
    view="ekyc-view-$role"
    if ! tmux has-session -t "=$view" 2>/dev/null; then
      tmux new-session -d -s "$view" -t "$session"
    fi
    tmux select-window -t "$view:$role"
    if [[ -z $(tmux list-clients -t "$view") ]]; then
      kitty --detach --title "eKYC Codex — $role" tmux attach-session -t "$view"
    fi
  done
  exit
fi
if [[ -n ${TMUX:-} ]]; then
  exec tmux switch-client -t "$session"
fi
exec tmux attach-session -t "$session"
