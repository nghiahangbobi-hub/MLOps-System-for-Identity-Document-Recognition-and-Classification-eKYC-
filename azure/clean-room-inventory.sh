#!/usr/bin/env bash
# Read-only inventory. Run from any directory with Azure CLI + ML v2 installed.
set -euo pipefail
umask 077

: "${AZ_SUBSCRIPTION:?Set approved subscription ID}"
: "${AZ_RESOURCE_GROUP:?Set approved resource group}"
: "${AZ_WORKSPACE:?Set approved workspace}"
az_bin=${AZ_BIN:-az}
root=${XDG_DATA_HOME:-"$HOME/.local/share"}/mlops-ekyc-review
root=$(realpath -m -- "$root")
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
case "$root/" in "$repo/"*) echo "Inventory must be outside the repository" >&2; exit 2;; esac
mkdir -p -- "$root"
record=$(mktemp -d "$root/azure-clean-room.XXXXXX")
printf '%s\n' "$record"
scope=(--subscription "$AZ_SUBSCRIPTION" --resource-group "$AZ_RESOURCE_GROUP" --workspace-name "$AZ_WORKSPACE")
export AZURE_EXTENSION_USE_DYNAMIC_INSTALL=no
date -u +%FT%TZ > "$record/started-utc.txt"
failed=0

capture() {
    local file=$1
    shift
    if "$az_bin" "$@" --output json > "$record/$file.tmp" 2> "$record/$file.stderr"; then
        mv -- "$record/$file.tmp" "$record/$file.json"
        printf 'PASS %s\n' "$file" >> "$record/status.txt"
    else
        printf 'FAILED %s\n' "$file" >> "$record/status.txt"
        failed=1
    fi
}

capture versions version
capture workspace ml workspace show --subscription "$AZ_SUBSCRIPTION" -g "$AZ_RESOURCE_GROUP" -n "$AZ_WORKSPACE" \
    --query '{name:name,location:location,storage_account:storage_account,id:id}'
capture jobs ml job list "${scope[@]}" --all-results true \
    --query '[].{name:name,status:status,experiment:experiment_name,compute:compute}'
capture archived-jobs ml job list "${scope[@]}" --all-results true --archived-only \
    --query '[].{name:name,status:status,experiment:experiment_name}'
capture computes ml compute list "${scope[@]}"
capture endpoints ml online-endpoint list "${scope[@]}"
capture datastores ml datastore list "${scope[@]}" \
    --query '[].{name:name,type:type,account:account_name,container:container_name}'
capture data-assets ml data list "${scope[@]}" --query '[].{name:name,version:version,type:type}'
capture models ml model list "${scope[@]}" --query '[].{name:name,version:version,type:type}'
capture environments ml environment list "${scope[@]}" --query '[].{name:name,version:version}'
if [[ -f "$record/workspace.json" ]]; then
    region=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["location"])' "$record/workspace.json")
    capture quota ml compute list-usage "${scope[@]}" --location "$region"
fi
date -u +%FT%TZ > "$record/finished-utc.txt"
cat "$record/status.txt" >&2
exit "$failed"
