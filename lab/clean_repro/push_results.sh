#!/bin/bash
# Push the small result files of <OUT> (json, txt, log tails) to branch lab-results/<NAME>.
# Uses a temporary worktree of the fork checkout, so the fork's own checkout never changes.
# Usage: bash push_results.sh <OUT> <NAME> <FORK>
OUT=$1; NAME=$2; FORK=$3
B=lab-results/$NAME
W=$(mktemp -d /tmp/push.XXXXXX)
cd "$FORK" || exit 0
git worktree add --detach "$W" HEAD >/dev/null 2>&1 || exit 0
(
  cd "$W" && git switch --orphan "tmp-$NAME-$$" >/dev/null 2>&1
  mkdir -p r
  (cd "$OUT" && find . \( -name '*.json' -o -name 'verdict.txt' -o -name 'status.txt' -o -name 'versions.txt' -o -name 'setup.txt' \) -size -2M) |
    while read -r f; do mkdir -p "r/$(dirname "$f")"; cp "$OUT/$f" "r/$f"; done
  for f in $(cd "$OUT" && find . -name '*.log'); do mkdir -p "r/$(dirname "$f")"; tail -n 80 "$OUT/$f" > "r/$f.tail.txt"; done
  for f in $(cd "$OUT" && find . -name 'metrics.jsonl'); do mkdir -p "r/$(dirname "$f")"; tail -n 60 "$OUT/$f" > "r/$f.tail"; done
  git add r >/dev/null 2>&1
  git -c user.name=Cb-dotcom -c user.email=charbel.elbeaini@lau.edu commit -qm "$NAME: $(tail -n 1 "$OUT/status.txt" 2>/dev/null)" >/dev/null 2>&1
  GIT_TERMINAL_PROMPT=0 git push -qf origin "HEAD:refs/heads/$B" >> "$OUT/push.log" 2>&1
  echo "$(date -u +%FT%TZ) push rc=$?" >> "$OUT/push.log"
)
git worktree remove --force "$W" >/dev/null 2>&1
git branch -D "tmp-$NAME-$$" >/dev/null 2>&1
exit 0
