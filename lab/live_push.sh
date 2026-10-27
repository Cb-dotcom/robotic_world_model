#!/bin/bash
# Push cmd30 results to branch lab-results/cmd30-live after every finished evaluation.
cd /workspace/Uncertainty-Aware-Robotic-World-Model-Go2/upstream/robotic_world_model
D=$(cat logs/p2/latest_cmd30.txt); B=lab-results/cmd30-live; last=-1
for i in $(seq 1 300); do
  n=$(grep -c rc= "$D/status.txt" 2>/dev/null || echo 0)
  if [ "$n" != "$last" ]; then
    /isaac-sim/python.sh "$D/partial_verdict.py" "$D" > "$D/verdict_live.txt" 2>&1
    W=$(mktemp -d /tmp/live.XXXXXX); git worktree add --detach "$W" HEAD >/dev/null 2>&1
    ( cd "$W" && git switch --orphan "tmp-live-$i" >/dev/null 2>&1; mkdir -p cmd30
      cp "$D/verdict_live.txt" cmd30/verdict.txt; cp "$D/status.txt" cmd30/ 2>/dev/null
      for x in 1601_42 1601_43 ceiling_42 ceiling_43; do cp "$D/$x/episodes_summary.json" "cmd30/$x.json" 2>/dev/null; done
      git add cmd30; git -c user.name=Cb-dotcom -c user.email=charbel.elbeaini@lau.edu commit -qm "cmd30 live: $n of 4 evaluations"
      GIT_TERMINAL_PROMPT=0 git push -qf origin HEAD:refs/heads/$B >> "$D/live_push.log" 2>&1
      echo "$(date -u +%FT%TZ) n=$n push rc=$?" >> "$D/live_push.log" )
    git worktree remove --force "$W"; git branch -D "tmp-live-$i" >/dev/null 2>&1; last=$n
  fi
  [ "$n" -ge 4 ] && break; sleep 60
done
