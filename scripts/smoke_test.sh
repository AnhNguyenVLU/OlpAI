#!/usr/bin/env bash
# Kiểm tra nhanh mọi bài: giải nén public_data.zip -> baseline.py -> score.py (split public).
#
#   scripts/smoke_test.sh             # dùng dữ liệu phát hành (chạy được trên repo thí sinh, dùng trong CI)
#   scripts/smoke_test.sh --teacher   # bộ giáo viên: sinh dữ liệu nhỏ + tập ẩn demo, chấm cả split private
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-public}"

status=0
for task in "$ROOT"/exams/*/bai-*/; do
  name="${task#"$ROOT"/exams/}"
  name="${name%/}"
  tmp="$(mktemp -d)"
  if (
    cd "$task"
    if [[ "$MODE" == "--teacher" ]]; then
      python generate_data.py --out "$tmp/data" --small --with-private >/dev/null
      python baseline.py --data "$tmp/data" --split private --out "$tmp/private.csv" >/dev/null
      python score.py --data "$tmp/data" --split private --pred "$tmp/private.csv" >/dev/null
    else
      unzip -q public_data.zip -d "$tmp"
    fi
    python baseline.py --data "$tmp/data" --out "$tmp/submission.csv" >/dev/null
    python score.py --data "$tmp/data" --pred "$tmp/submission.csv"
  ) >"$tmp/log" 2>&1; then
    printf '✔ %-55s %s\n' "$name" "$(tail -n1 "$tmp/log")"
  else
    printf '✘ %s\n' "$name"; sed 's/^/    /' "$tmp/log"; status=1
  fi
  rm -rf "$tmp"
done
exit $status
