#!/bin/bash
# Plan v6 §10 housekeeping: pack the E1 results for an off-workstation backup, with a sha256 manifest.
# Run from the robotic_world_model root inside the container (CPU only, read-only on logs/):
#   scripts/reinforcement_learning/model_based/backup_e1_results.sh <fresh_out_dir> [--no-data]
# Then copy <out_dir> off the workstation (scp on the lab PC or the laptop, never in the container) and
# run `sha256sum -c e1_results.tar.sha256` and `tar -xOf e1_results.tar MANIFEST.sha256 | head` there.
#
# Packed (paths relative to the repo root):
#   logs/p2/{e1A_1111_1650,e1v_1113_0939,e1v_fb_1113_1713,e1r_1113_1853}: summaries, status, queue logs,
#     run_args/json, match_*.txt, git.txt, inputs.sha256, hash_check, model_check/, score*/, eval logs,
#     episodes.csv, world models (*.pt: e1v_wm.pt and feedback updates)
#   E1 policies: final policy_*.pt + run_args.json + params/ of runs 1601 1701 1901 2001 2003 2004 2102 2103 2402 2403
#   unless --no-data: the fall data of the RQ1 battery (record/*.npz, batch_*.npz, data/ of the E1-V folders)
set -uo pipefail
OUT=${1:?fresh output dir}
DATA=1; [ "${2:-}" = "--no-data" ] && DATA=0
RUNS="1601 1701 1901 2001 2003 2004 2102 2103 2402 2403"
P2="logs/p2/e1A_1111_1650 logs/p2/e1v_1113_0939 logs/p2/e1v_fb_1113_1713 logs/p2/e1r_1113_1853"
[ -e "$OUT" ] && { echo "OUT exists: $OUT"; exit 2; }
mkdir -p "$OUT"
LIST="$OUT/files.txt"; : > "$LIST"
for d in $P2; do
  [ -d "$d" ] || { echo "missing folder $d"; exit 1; }
  find "$d" -type f \( -name 'summary*' -o -name 'status_*.txt' -o -name 'queue.log' -o -name '*.json' \
    -o -name 'match_*.txt' -o -name 'git*.txt' -o -name 'inputs.sha256' -o -name 'hash_check.txt' \
    -o -name 'episodes.csv' -o -name 'eval.log' -o -name '*.pt' -o -name '*.nohup' -o -name 'report.txt' \) >> "$LIST"
  find "$d" -type f \( -path '*/model_check/*' -o -path '*/score*/*' \) >> "$LIST"
  if [ $DATA -eq 1 ]; then
    find "$d" -type f \( -path '*/record/*.npz' -o -name 'batch_*.npz' -o -path '*/data/*' -o -path '*/data_*/*' \) >> "$LIST"
  fi
done
for r in $RUNS; do
  dirs=$(ls -d logs/offline/anymal_d_flat/*_"$r" 2>/dev/null)
  [ -n "$dirs" ] || { echo "no run folder for $r"; exit 1; }
  for rd in $dirs; do
    last=$(ls "$rd"/policy_*.pt 2>/dev/null | sed 's/.*policy_\([0-9]*\)\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2)
    [ -n "$last" ] || { echo "no policy in $rd"; exit 1; }
    echo "$last" >> "$LIST"
    [ -f "$rd/run_args.json" ] && echo "$rd/run_args.json" >> "$LIST"
    [ -d "$rd/params" ] && find "$rd/params" -type f >> "$LIST"
  done
done
sort -u "$LIST" -o "$LIST"
xargs -d '\n' sha256sum < "$LIST" > "$OUT/MANIFEST.sha256" || { echo "sha256 failed"; exit 1; }
{ git rev-parse HEAD; date -u +%FT%TZ; echo "files $(wc -l < "$LIST")"; } > "$OUT/BACKUP_INFO.txt"
tar -cf "$OUT/e1_results.tar" -C "$OUT" MANIFEST.sha256 BACKUP_INFO.txt -C "$PWD" -T "$LIST" || { echo "tar failed"; exit 1; }
( cd "$OUT" && sha256sum e1_results.tar > e1_results.tar.sha256 )
echo "files: $(wc -l < "$LIST"); size: $(du -h "$OUT/e1_results.tar" | cut -f1)"
awk '{print $2}' "$OUT/MANIFEST.sha256" | sed 's|^\(logs/[^/]*/[^/]*\)/.*|\1|' | sort | uniq -c
cat "$OUT/e1_results.tar.sha256"
