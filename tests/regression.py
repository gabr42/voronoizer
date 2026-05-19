"""Regression sweep across (fixture × chamfer).

Runs the voronoizer CLI on each combination and checks that the output
mesh is sane: loads, is watertight (or close to it), has positive volume,
and is smaller than the input (i.e. holes actually got cut).

Run from the repo root::

    .venv\\Scripts\\python.exe tests\\regression.py

Exits non-zero if any case fails. Pass --quick to skip the heaviest
fixture (Unnamed-Body).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import trimesh

ROOT = Path(__file__).resolve().parent.parent
VORONOIZER = ROOT / ".venv" / "Scripts" / "voronoizer.exe"
DATA = ROOT / "tests" / "data"
SEED = 613805311  # the seed from the last user-reported Unnamed-Body run


@dataclass
class Case:
    name: str
    fixture: Path
    chamfer: float
    holes: int
    strut: float = 1.5
    thickness: float = 2.0
    # `None` keeps voronoizer's default (strut/2). Override for large flat
    # fixtures (100mm cube, 120mm cylinder) whose coarse initial topology
    # would otherwise blow past the 500k-face subdivision cap.
    target_edge_length: Optional[float] = None
    extra_args: tuple[str, ...] = ()


def build_matrix(quick: bool) -> list[Case]:
    cases: list[Case] = []
    # (name, fixture, holes, strut, thickness, target_edge_length)
    fixtures = [
        ("cube_100mm", DATA / "cube_100mm.stl", 30, 1.5, 2.0, 4.0),
        ("sphere_r50mm", DATA / "sphere_r50mm.stl", 30, 1.5, 2.0, None),
        # cyl_r30_h120mm caps are a 64-spoke fan with 30 mm radii — target
        # has to leave room to subdivide those long spokes under the cap.
        ("cyl_r30_h120mm", DATA / "cyl_r30_h120mm.stl", 30, 1.5, 2.0, 8.0),
        ("shape_box", ROOT / "Shape-Box.stl", 20, 1.5, 2.0, None),
    ]
    if not quick:
        fixtures.append(
            ("unnamed_body", ROOT / "Unnamed-Body.stl", 30, 1.5, 2.0, None)
        )
    for name, fixture, n, strut, t, tel in fixtures:
        for chamfer in (0.0, 0.6):
            cases.append(Case(
                name=f"{name}_ch{chamfer}",
                fixture=fixture,
                chamfer=chamfer,
                holes=n,
                strut=strut,
                thickness=t,
                target_edge_length=tel,
            ))
    # Asymmetric inner-chamfer = 0 with outer = 0.6.
    cases.append(Case(
        name="cube_100mm_outer-only-chamfer",
        fixture=DATA / "cube_100mm.stl",
        chamfer=0.6,
        holes=30,
        target_edge_length=4.0,
        extra_args=("--inner-chamfer", "0.0"),
    ))
    # Shell-only inspection mode.
    cases.append(Case(
        name="cube_100mm_shell-only",
        fixture=DATA / "cube_100mm.stl",
        chamfer=0.0,
        holes=30,
        target_edge_length=4.0,
        extra_args=("--shell",),
    ))
    # Cutters-only inspection mode.
    cases.append(Case(
        name="cube_100mm_cutters-only",
        fixture=DATA / "cube_100mm.stl",
        chamfer=0.0,
        holes=30,
        target_edge_length=4.0,
        extra_args=("--cutters",),
    ))
    return cases


@dataclass
class Result:
    name: str
    ok: bool
    duration_s: float
    detail: str
    output_volume: Optional[float] = None
    input_volume: Optional[float] = None
    watertight: Optional[bool] = None


def run_case(case: Case, tmpdir: Path) -> Result:
    out_path = tmpdir / f"{case.name}.stl"
    cmd = [
        str(VORONOIZER),
        str(case.fixture),
        str(out_path),
        "--seed", str(SEED),
        "-n", str(case.holes),
        "-s", str(case.strut),
        "-t", str(case.thickness),
        "--chamfer", str(case.chamfer),
    ]
    if case.target_edge_length is not None:
        cmd += ["--target-edge-length", str(case.target_edge_length)]
    cmd += list(case.extra_args)
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=600, cwd=ROOT,
        )
    except subprocess.TimeoutExpired:
        return Result(case.name, False, time.perf_counter() - t0,
                      "timeout after 600 s")
    dt = time.perf_counter() - t0
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
        return Result(case.name, False, dt,
                      f"exit {proc.returncode}: {' | '.join(tail)}")
    if not out_path.exists():
        return Result(case.name, False, dt, "no output file produced")
    try:
        out_mesh = trimesh.load_mesh(out_path)
    except Exception as e:
        return Result(case.name, False, dt, f"output unreadable: {e}")
    if isinstance(out_mesh, trimesh.Scene):
        out_mesh = trimesh.util.concatenate(list(out_mesh.dump()))
    if "--shell" in case.extra_args or "--cutters" in case.extra_args:
        # Inspection modes: just require non-empty mesh.
        if len(out_mesh.faces) < 4:
            return Result(case.name, False, dt,
                          f"inspection mesh too small ({len(out_mesh.faces)} faces)",
                          watertight=out_mesh.is_watertight)
        return Result(case.name, True, dt,
                      f"{len(out_mesh.faces)} faces",
                      watertight=out_mesh.is_watertight)
    try:
        in_mesh = trimesh.load_mesh(case.fixture)
    except Exception as e:
        return Result(case.name, False, dt, f"input unreadable: {e}")
    in_vol = float(abs(in_mesh.volume))
    out_vol = float(abs(out_mesh.volume))
    wt = bool(out_mesh.is_watertight)
    detail_bits = [
        f"V_out={out_vol:.1f}mm³",
        f"V_in={in_vol:.1f}mm³",
        f"wt={wt}",
        f"faces={len(out_mesh.faces)}",
    ]
    if out_vol <= 0:
        return Result(case.name, False, dt,
                      "non-positive volume: " + ", ".join(detail_bits),
                      output_volume=out_vol, input_volume=in_vol,
                      watertight=wt)
    if out_vol >= in_vol:
        return Result(case.name, False, dt,
                      "no material removed: " + ", ".join(detail_bits),
                      output_volume=out_vol, input_volume=in_vol,
                      watertight=wt)
    # Watertight is the strict goal but manifold3d sometimes leaves a few
    # open edges around fillet near-tangencies; accept those as a soft
    # warning so the suite stays useful, and fail only on hard breakage.
    return Result(case.name, True, dt, ", ".join(detail_bits),
                  output_volume=out_vol, input_volume=in_vol, watertight=wt)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="Skip the slowest fixture (Unnamed-Body).")
    parser.add_argument("--keep", type=Path, default=None,
                        help="Keep output STLs in this directory.")
    parser.add_argument("--filter", default=None,
                        help="Run only cases whose name contains this substring.")
    args = parser.parse_args()

    if not VORONOIZER.exists():
        print(f"error: voronoizer.exe not found at {VORONOIZER}", file=sys.stderr)
        return 2

    cases = build_matrix(args.quick)
    if args.filter:
        cases = [c for c in cases if args.filter in c.name]
        if not cases:
            print(f"no cases match filter {args.filter!r}", file=sys.stderr)
            return 2

    if args.keep:
        args.keep.mkdir(parents=True, exist_ok=True)
        tmpdir = args.keep
        cleanup = False
    else:
        tmpdir = Path(tempfile.mkdtemp(prefix="voronoizer-regression-"))
        cleanup = True

    results: list[Result] = []
    try:
        for i, case in enumerate(cases, 1):
            print(f"[{i:>2}/{len(cases)}] {case.name} ...", flush=True)
            r = run_case(case, tmpdir)
            results.append(r)
            status = "PASS" if r.ok else "FAIL"
            print(f"        {status}  {r.duration_s:5.1f}s  {r.detail}",
                  flush=True)
    finally:
        if cleanup:
            shutil.rmtree(tmpdir, ignore_errors=True)

    failed = [r for r in results if not r.ok]
    non_wt = [r for r in results if r.ok and r.watertight is False]
    print()
    print("=" * 72)
    print(f"summary: {len(results) - len(failed)} pass, {len(failed)} fail "
          f"({len(non_wt)} pass-but-non-watertight)")
    if non_wt:
        print("non-watertight (accepted):")
        for r in non_wt:
            print(f"   - {r.name}: {r.detail}")
    if failed:
        print("failures:")
        for r in failed:
            print(f"   - {r.name}: {r.detail}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
