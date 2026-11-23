#!/usr/bin/env python3
"""A4 D6 (22 Nov 2026, Charbel's decision 20:47): D2's pooled data with real startup prefixes.

Same two recordings and the same E22 split as D2 (go2_mixed_prepare): the nominal s2 round-0 recording and the D1
fitting batch (s2 at 0.30 friction), 10 % of whole episodes withheld with seed 101 (go2_round0_prepare.episodes /
withheld_set; the withheld lists must equal D2's). Difference to D2: every training episode that began at a
physical reset (recorded episode_step == 1 at its first row) gets ONE real row in front of it:

    reset row = [state[first] (the recorder's pre-action state of the episode's first step, 45),
                 previous_action[first] (12), contact 0 (8), termination 0 (1)]

The fitter pairs state row j with action row j + 1, so the reset row with the next row's action is the first real
transition (reset state, first action) -> first recorded successor. No copies, no padding.

Output (fresh folder):
  segments/      every training episode with >= 2 rows after the prefix (r0_ / sh_ names, fitter layout,
                 manifest.csv); episodes shorter than the 40-row fitter window are kept here for startup windows
                 (the fitter skips them for standard windows with --skip_short_segments)
  startup.json   the segment names whose first row is a physical reset row (the fitter's startup windows)
  start_data/    per-env blocks (same rows, reset rows included; episode-end flag) for warm imagination starts
  reset_states.npz  state (raw 45) and previous_action (raw 12) of every training reset row, plus source/env/row
  manifest.json  counts per source (episodes, training, withheld, with reset row, short, falls, falls with a
                 terminal target in standard windows / startup windows), mean training-episode length,
                 startup_reset_share q = min(0.5, 256 / mean length), hashes, the rule
The normalizer is not computed here: D6 uses D2's frozen go2_norm.npz.

Reset-row torque (23 Nov 2026, D6b): the recorder's pre-action torque block (state dims 33..44) after an earlier episode
is that episode's last torque (the simulator has not recomputed it). reset_torque="zero" (D6b) writes 0 N m there, in
the segments and in reset_states.npz: before the first action no actuator torque has been applied. "recorded" (D6)
keeps the recorded value.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

MB = Path(__file__).resolve().parent
for p in (str(MB), str(MB / "analysis")):
    if p not in sys.path:
        sys.path.insert(0, p)

import go2_round0_prepare as prep  # noqa: E402
from pre_reset_transitions import sha256, validate_trace  # noqa: E402

IMAGINED_EPISODE_STEPS = 256
Q_CAP = 0.5


TORQUE = slice(33, 45)
RESET_TORQUE = ("recorded", "zero")


def reset_state(trace, n, s, reset_torque="recorded"):
    st = trace["state"][s, n].astype(np.float32).copy()
    if reset_torque == "zero":
        st[TORQUE] = 0.0
    elif reset_torque != "recorded":
        raise ValueError(f"reset_torque must be one of {RESET_TORQUE}")
    return st


def reset_row(trace, n, s, reset_torque="recorded"):
    return np.concatenate([reset_state(trace, n, s, reset_torque), trace["previous_action"][s, n],
                           np.zeros(prep.CONTACT, np.float32), np.zeros(prep.TERM, np.float32)]).astype(np.float32)


def prepare_source(trace_path, out, tag, frac=0.1, seed=101, reset_torque="recorded"):
    """One source: segments, start blocks, reset states and withheld.json in `out` (fresh)."""
    out = Path(out)
    out.mkdir(parents=True)
    with np.load(trace_path, allow_pickle=False) as z:
        meta = validate_trace(z)
        trace = {k: z[k] for k in z.files if k != "metadata"}
    for k in ("state", "previous_action", "episode_step"):
        if k not in trace:
            raise ValueError(f"{trace_path}: the recording has no '{k}' field")
    seq_len = prep.seq_len_from_config()
    term, tout, tob, step = trace["terminated"], trace["time_out"], trace["termination_observation"], trace["episode_step"]
    eps = prep.episodes(trace["done"])
    held = prep.withheld_set(eps, frac, seed)
    seg_dir = out / "segments"
    seg_dir.mkdir()
    c = dict(episodes=len(eps), falls=0, withheld=0, withheld_falls=0, training=0, training_falls=0,
             with_reset_row=0, without_reset_row=0, too_short=0, short_segments=0, segments=0, segment_rows=0,
             falls_in_standard_windows=0, falls_in_startup_windows=0, training_episode_rows=0)
    manifest, startup, held_list, blocks, resets = [], [], [], {}, []
    for i, (n, s, e, finished) in enumerate(eps):
        fall = bool(finished and term[e, n] and not tout[e, n])
        kind = "fall" if fall else ("time_out" if finished else "unfinished")
        c["falls"] += fall
        if tob[s:e, n].sum() != 0:
            raise ValueError(f"env {n}: termination label before the episode end")
        if i in held:
            c["withheld"] += 1
            c["withheld_falls"] += fall
            held_list.append(dict(env=n, first=s, last=e, kind=kind))
            continue
        c["training"] += 1
        c["training_falls"] += fall
        c["training_episode_rows"] += e - s + 1
        rows = prep.rows_of(trace, n, s, e)
        has_reset = int(step[s, n]) == 1
        if has_reset:
            rows = np.concatenate([reset_row(trace, n, s, reset_torque)[None], rows], axis=0)
            resets.append((reset_state(trace, n, s, reset_torque), trace["previous_action"][s, n], n, s))
            c["with_reset_row"] += 1
        else:
            c["without_reset_row"] += 1
        block = rows.copy()
        block[:, -1] = 0.0
        block[-1, -1] = 1.0                          # episode end: the start sampler's boundary
        blocks.setdefault(n, []).append(block)
        if len(rows) < 2:
            c["too_short"] += 1
            continue
        name = f"{tag}_env{n:03d}_r{s:05d}"
        np.savetxt(seg_dir / f"{name}.csv", rows, delimiter=",", fmt="%.9g")
        manifest.append((name, len(rows)))
        c["segments"] += 1
        c["segment_rows"] += len(rows)
        if len(rows) < seq_len:
            c["short_segments"] += 1
        if has_reset:
            startup.append(name)
        c["falls_in_standard_windows"] += fall and len(rows) >= seq_len
        c["falls_in_startup_windows"] += fall and has_reset
    (seg_dir / "manifest.csv").write_text("name,steps,rows,num_envs\n" + "".join(f"{a},{b},{b},1\n" for a, b in manifest))
    start_dir = out / "start_data"
    start_dir.mkdir()
    files = 0
    for n in sorted(blocks):
        np.savetxt(start_dir / f"state_action_data_{files}.csv", np.concatenate(blocks[n]), delimiter=",", fmt="%.9g")
        files += 1
    (out / "withheld.json").write_text(json.dumps(held_list, indent=1))
    st = np.stack([r[0] for r in resets]).astype(np.float32) if resets else np.zeros((0, prep.STATE), np.float32)
    pa = np.stack([r[1] for r in resets]).astype(np.float32) if resets else np.zeros((0, prep.ACTION), np.float32)
    return dict(source=str(trace_path), source_sha256=sha256(trace_path), checkpoint_sha256=meta.get("checkpoint_sha256"),
                counts=c, startup=startup, manifest=manifest, files=files,
                resets=dict(state=st, previous_action=pa, env=np.array([r[2] for r in resets], np.int32),
                            row=np.array([r[3] for r in resets], np.int32)))


def prepare_startup(r0_trace, shift_trace, out, frac=0.1, seed=101, d2_data=None, reset_torque="recorded"):
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"{out} exists")
    out.mkdir(parents=True)
    if reset_torque not in RESET_TORQUE:
        raise ValueError(f"reset_torque must be one of {RESET_TORQUE}")
    src = {"r0": prepare_source(r0_trace, out / "r0", "r0", frac, seed, reset_torque),
           "sh": prepare_source(shift_trace, out / "shift", "sh", frac, seed, reset_torque)}
    if d2_data is not None:
        for tag, sub in (("r0", "r0"), ("sh", "shift")):
            if (out / sub / "withheld.json").read_text() != (Path(d2_data) / sub / "withheld.json").read_text():
                raise ValueError(f"{sub}: the withheld episodes differ from D2's")
    seg = out / "segments"
    seg.mkdir()
    lines, startup = ["name,steps,rows,num_envs"], []
    for tag, sub in (("r0", "r0"), ("sh", "shift")):
        for name, n in src[tag]["manifest"]:
            (seg / f"{name}.csv").write_bytes((out / sub / "segments" / f"{name}.csv").read_bytes())
            lines.append(f"{name},{n},{n},1")
        startup += src[tag]["startup"]
    (seg / "manifest.csv").write_text("\n".join(lines) + "\n")
    (out / "startup.json").write_text(json.dumps(dict(segments=startup, rule="first row = physical reset row"), indent=1))
    start = out / "start_data"
    start.mkdir()
    files, rows, min_rows = 0, 0, None
    for sub in ("r0", "shift"):
        for f in sorted((out / sub / "start_data").glob("state_action_data_*.csv"), key=lambda p: int(p.stem.rsplit("_", 1)[1])):
            data = f.read_bytes()
            (start / f"state_action_data_{files}.csv").write_bytes(data)
            n = data.count(b"\n")
            rows += n
            min_rows = n if min_rows is None else min(min_rows, n)
            files += 1
    r = {k: np.concatenate([src["r0"]["resets"][k], src["sh"]["resets"][k]]) for k in ("state", "previous_action", "env", "row")}
    r["source"] = np.array([0] * len(src["r0"]["resets"]["env"]) + [1] * len(src["sh"]["resets"]["env"]), np.int8)
    with (out / "reset_states.npz").open("xb") as f:
        np.savez(f, **r)
    episodes = sum(s["counts"]["training"] for s in src.values())
    ep_rows = sum(s["counts"]["training_episode_rows"] for s in src.values())
    mean_len = ep_rows / max(episodes, 1)
    q = min(Q_CAP, IMAGINED_EPISODE_STEPS / mean_len)
    (start / "start_data.json").write_text(json.dumps(dict(
        rows=rows, files=files, min_file_rows=int(min_rows), episodes=episodes,
        layout="next_state 45 (a reset row holds the pre-action reset state), action 12, contact 8, episode-end flag 1",
        sources="round-0 nominal + D1 shift (0.30 friction), pooled, with startup prefixes (A4 D6)"), indent=1))
    result = dict(
        rule=("A4 D6: D2's sources and E22 split (fraction %s, seed %s); each training episode that began at a physical "
              "reset is prefixed by one real row [pre-action reset state, previous action, contact 0, termination 0]; "
              "no padding; D2's normalizer is used unchanged" % (frac, seed)),
        sources={t: dict(path=s["source"], sha256=s["source_sha256"], checkpoint_sha256=s["checkpoint_sha256"],
                         counts=s["counts"]) for t, s in src.items()},
        reset_torque=reset_torque, segments=len(lines) - 1, startup_segments=len(startup), reset_states=int(len(r["env"])),
        mean_training_episode_rows=mean_len, startup_reset_share=q,
        startup_reset_share_rule=f"q = min({Q_CAP}, {IMAGINED_EPISODE_STEPS} / mean training-episode rows)",
        outputs=dict(segments_manifest_sha256=sha256(seg / "manifest.csv"), startup_json_sha256=sha256(out / "startup.json"),
                     reset_states_sha256=sha256(out / "reset_states.npz"),
                     start_data_json_sha256=sha256(start / "start_data.json"),
                     r0_withheld_sha256=sha256(out / "r0" / "withheld.json"),
                     shift_withheld_sha256=sha256(out / "shift" / "withheld.json")),
        code_sha256=sha256(__file__))
    (out / "manifest.json").write_text(json.dumps(result, indent=1))
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--round0", type=Path, required=True)
    p.add_argument("--shift", type=Path, required=True, help="the D1 fitting batch")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--d2-data", type=Path, default=None, help="D2's data folder: the withheld lists must be equal")
    p.add_argument("--reset-torque", choices=RESET_TORQUE, default="recorded",
                   help="reset-row torque block: as recorded (D6) or 0 N m (D6b)")
    a = p.parse_args(argv)
    r = prepare_startup(a.round0, a.shift, a.out, d2_data=a.d2_data, reset_torque=a.reset_torque)
    for t, s in r["sources"].items():
        c = s["counts"]
        print(f"[d6] {t}: {c['training']} training episodes ({c['training_falls']} falls), {c['with_reset_row']} with a "
              f"reset row, {c['without_reset_row']} without; segments {c['segments']} ({c['short_segments']} shorter than "
              f"the 40-row window); falls with a terminal target: standard windows {c['falls_in_standard_windows']}, "
              f"startup windows {c['falls_in_startup_windows']}")
    print(f"[d6] reset-row torque: {r['reset_torque']}; {r['segments']} segments, {r['startup_segments']} with a reset row; reset states {r['reset_states']}; "
          f"mean training episode {r['mean_training_episode_rows']:.1f} rows -> startup reset share q = "
          f"{r['startup_reset_share']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
