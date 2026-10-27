import json, sys
from pathlib import Path
d = Path(sys.argv[1])
runs = ("1601_42", "1601_43", "ceiling_42", "ceiling_43")
got = {}
for x in runs:
    p = d / x / "episodes_summary.json"
    if p.is_file():
        try:
            got[x] = json.loads(p.read_text())["summary"]
        except Exception as e:
            print(x, "unreadable:", e)
st = (d / "status.txt").read_text().split("\n") if (d / "status.txt").is_file() else []
print("status:", [l for l in st if l])
for x in runs:
    if x in got:
        r = got[x]
        print(x, {k: r.get(k) for k in ("falls", "episodes_completed", "track_per_step", "speed_ratio", "time_to_fall_counts")})
    else:
        print(x, "not finished")
if all(x in got for x in runs):
    f1 = got["1601_42"]["falls"] + got["1601_43"]["falls"]; n1 = got["1601_42"]["episodes_completed"] + got["1601_43"]["episodes_completed"]
    fc = got["ceiling_42"]["falls"] + got["ceiling_43"]["falls"]; nc = got["ceiling_42"]["episodes_completed"] + got["ceiling_43"]["episodes_completed"]
    print(f"1601 pooled falls 30 s schedule: {f1}/{n1}  (10 s schedule: 156/689; rule: <= 78)")
    print(f"ceiling pooled falls 30 s schedule: {fc}/{nc}  (rule: <= 6)")
    print("VERDICT:", "TRANSITION-CONCENTRATED (rule met)" if (f1 <= 78 and fc <= 6) else "NOT transition-specific by the rule")
else:
    print(f"PARTIAL: {len(got)} of 4 evaluations finished")
