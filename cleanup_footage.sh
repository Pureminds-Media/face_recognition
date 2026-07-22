#!/usr/bin/env bash
# Delete footage files older than 7 days.

set -euo pipefail

FOOTAGE_DIR="/mnt/camera_system/footage"

if [[ ! -d "$FOOTAGE_DIR" ]]; then
    echo "ERROR: footage directory does not exist: $FOOTAGE_DIR" >&2
    exit 1
fi

if ! mountpoint -q "$(dirname "$FOOTAGE_DIR")"; then
    echo "ERROR: $(dirname "$FOOTAGE_DIR") is not a mounted filesystem — refusing to run" >&2
    exit 1
fi

DAYS="${FOOTAGE_RETENTION_DAYS:-7}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Cleaning footage older than ${DAYS} days in: $FOOTAGE_DIR"

deleted=0
freed=0

while IFS= read -r -d '' file; do
    size=$(stat -c%s "$file" 2>/dev/null || echo 0)
    rm -f "$file"
    freed=$((freed + size))
    deleted=$((deleted + 1))
done < <(find "$FOOTAGE_DIR" -maxdepth 1 -type f \
    \( -name "*.webm" -o -name "*.mp4" -o -name "*.mov" -o -name "*.avi" -o -name "*.mkv" -o -name "*.m4v" \) \
    -mtime +"$((DAYS - 1))" -print0)

freed_mb=$(echo "scale=1; $freed / 1048576" | bc)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done. Removed $deleted file(s), freed ~${freed_mb} MB."
