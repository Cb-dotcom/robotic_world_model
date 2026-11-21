"""A4 D1 (21 Nov 2026): a static ground-friction shift for Go2 recordings and evaluations, with read-back.

The shift scales the terrain (ground) material's static and dynamic friction by one factor. PhysX combines the two
materials in contact by the mode with the higher priority (average < min < multiply < max). The Isaac Lab velocity
terrain uses "multiply", so the effective robot-ground friction scales by the same factor unless a robot material
uses "max". `verify` reads the materials back from the USD stage after the environment is built and refuses to
continue if the effective scaling cannot be confirmed.
"""
from __future__ import annotations

PRIORITY = {"average": 0, "min": 1, "multiply": 2, "max": 3}


def combine(a, b, mode_a, mode_b):
    """PhysX friction combine for two materials (the higher-priority mode wins)."""
    mode = mode_a if PRIORITY[mode_a] >= PRIORITY[mode_b] else mode_b
    return {"average": (a + b) / 2.0, "min": min(a, b), "multiply": a * b, "max": max(a, b)}[mode], mode


def scale_ground(env_cfg, factor):
    """Scale the terrain material in the env config. Returns the nominal and scaled values."""
    if not 0.0 < factor <= 1.0:
        raise ValueError("friction factor must be in (0, 1]")
    mat = env_cfg.scene.terrain.physics_material
    nominal = dict(static=float(mat.static_friction), dynamic=float(mat.dynamic_friction),
                   combine_mode=str(mat.friction_combine_mode))
    mat.static_friction = nominal["static"] * factor
    mat.dynamic_friction = nominal["dynamic"] * factor
    return dict(factor=float(factor), terrain_prim=str(env_cfg.scene.terrain.prim_path), nominal=nominal,
                scaled=dict(static=float(mat.static_friction), dynamic=float(mat.dynamic_friction)))


def check(record, materials, tol=1e-5):
    """Pure check on read-back materials: [(path, static, dynamic, combine_mode)].

    Ground materials = those under the terrain prim. Confirms the ground values equal the scaled config and that the
    effective friction against every other material scales by `factor` (no "max" mode, ground mode wins).
    """
    root, f = record["terrain_prim"], record["factor"]
    ground = [m for m in materials if m[0].startswith(root)]
    others = [m for m in materials if not m[0].startswith(root)]
    if not ground:
        raise RuntimeError(f"no physics material found under {root}")
    for path, s, d, mode in ground:
        if abs(s - record["scaled"]["static"]) > tol or abs(d - record["scaled"]["dynamic"]) > tol:
            raise RuntimeError(f"ground material {path} reads {s:.4f}/{d:.4f}, expected the scaled config")
    g_s, g_d, g_mode = ground[0][1], ground[0][2], ground[0][3]
    n_s, n_d = record["nominal"]["static"], record["nominal"]["dynamic"]
    ratios = []
    for path, s, d, mode in others or [("(robot default)", 0.5, 0.5, "average")]:
        eff_s, used = combine(s, g_s, mode, g_mode)
        nom_s, _ = combine(s, n_s, mode, g_mode)
        eff_d, _ = combine(d, g_d, mode, g_mode)
        nom_d, _ = combine(d, n_d, mode, g_mode)
        ratios.append((path, used, eff_s / nom_s if nom_s else float("nan"), eff_d / nom_d if nom_d else float("nan")))
    bad = [r for r in ratios if abs(r[2] - f) > 1e-4 or abs(r[3] - f) > 1e-4]
    if bad:
        raise RuntimeError(f"effective friction does not scale by {f}: {bad[:3]}")
    return dict(ground=[dict(path=p, static=s, dynamic=d, combine_mode=m) for p, s, d, m in ground],
                other_materials=len(others), combine_modes_used=sorted({r[1] for r in ratios}),
                effective_ratio_static=f, effective_ratio_dynamic=f, verified=True)


def read_materials(env_prefix="/World/envs/env_0"):
    """All USD physics materials of the ground and of env 0 (Isaac Sim only)."""
    import omni.usd
    from pxr import PhysxSchema, UsdPhysics
    stage = omni.usd.get_context().get_stage()
    out = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not prim.HasAPI(UsdPhysics.MaterialAPI):
            continue
        if path.startswith("/World/envs/") and not path.startswith(env_prefix):
            continue
        api = UsdPhysics.MaterialAPI(prim)
        px = PhysxSchema.PhysxMaterialAPI(prim) if prim.HasAPI(PhysxSchema.PhysxMaterialAPI) else None
        mode = (px.GetFrictionCombineModeAttr().Get() if px else None) or "average"
        out.append((path, float(api.GetStaticFrictionAttr().Get() or 0.0), float(api.GetDynamicFrictionAttr().Get() or 0.0),
                    str(mode)))
    return out


def verify(record, robot=None):
    """Read back after the environment is built; also report the robot's PhysX shape materials."""
    rec = check(record, read_materials())
    if robot is not None:
        try:
            m = robot.root_physx_view.get_material_properties()
            rec["robot_physx_material_mean"] = dict(static=float(m[..., 0].mean()), dynamic=float(m[..., 1].mean()))
        except Exception as error:  # reported, not fatal: the USD check above is the gate
            rec["robot_physx_material_mean"] = f"unavailable: {error}"
    return rec
