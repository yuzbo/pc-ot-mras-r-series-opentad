#!/usr/bin/env bash
set -euo pipefail

WAIT_SCREEN="${WAIT_SCREEN:?WAIT_SCREEN is required}"
APPROVAL_FILE="${APPROVAL_FILE:?APPROVAL_FILE is required}"
POLL_SECONDS="${POLL_SECONDS:-300}"
RUN_LABEL="${RUN_LABEL:-queued_run}"

if [[ "$#" -lt 1 ]]; then
  echo "Usage: WAIT_SCREEN=<screen> APPROVAL_FILE=<path> $0 <command> [args...]" >&2
  exit 2
fi

stamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

while screen -ls | grep -q "[.]${WAIT_SCREEN}[[:space:]]"; do
  echo "$(stamp) waiting for screen ${WAIT_SCREEN}"
  sleep "$POLL_SECONDS"
done

echo "$(stamp) wait screen ${WAIT_SCREEN} is absent"

while [[ ! -f "$APPROVAL_FILE" ]]; do
  echo "$(stamp) waiting for gate approval file ${APPROVAL_FILE}"
  sleep "$POLL_SECONDS"
done

echo "$(stamp) gate approval found for ${RUN_LABEL}: ${APPROVAL_FILE}"
echo "$(stamp) launching ${RUN_LABEL}: $*"
exec "$@"
