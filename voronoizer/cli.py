"""CLI entrypoint for voronoizer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from voronoizer import __version__
from voronoizer import progress


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="voronoizer",
        description="Turn an STL file into a Voronoi-perforated shell.",
    )
    p.add_argument("input", type=Path, help="Input STL file.")
    p.add_argument("output", type=Path, help="Output STL file.")

    p.add_argument(
        "-t", "--shell-thickness",
        type=float, default=2.0,
        help="Shell wall thickness in millimeters (default: %(default)s).",
    )
    p.add_argument(
        "-n", "--holes",
        type=int, default=100,
        help="Approximate hole count (default: %(default)s).",
    )
    p.add_argument(
        "-s", "--strut-thickness",
        type=float, default=1.5,
        help="Width of strut between holes in mm (default: %(default)s).",
    )

    p.add_argument(
        "--top-bottom-only",
        action="store_true",
        help="Perforate only top/bottom faces.",
    )
    p.add_argument(
        "--normal-angle",
        type=float, default=30.0,
        help="Max angle (deg) from +/-Z that counts as top/bottom (default: %(default)s).",
    )

    p.add_argument(
        "--edge-margin",
        type=float, default=None,
        help="Min distance (mm) seeds must keep from sharp mesh edges. "
             "Default: auto-scaled from seed density.",
    )
    p.add_argument(
        "--chamfer",
        type=float, default=0.0,
        help="Chamfer distance (mm) for hole edges where they meet the shell "
             "surfaces. 0 = no chamfer (default). Clamped to safe maximum.",
    )
    p.add_argument(
        "--soft-edge-angle",
        type=float, default=25.0,
        help="Dihedral angle (deg) above which mesh edges count as 'sharp' "
             "for seed-rejection. Default 25 deg works for boxy / faceted "
             "models with crisp edges. Lower it (e.g. 10) for CAD-like "
             "models with smooth fillets where holes near the fillets would "
             "otherwise look chewed up.",
    )

    p.add_argument(
        "--seed",
        type=int, default=None,
        help="Random seed for reproducible patterns.",
    )
    p.add_argument(
        "--repair",
        action="store_true",
        help="Best-effort repair of non-manifold / non-watertight input.",
    )

    diag = p.add_argument_group(
        "Diagnostic / inspection modes (mutually exclusive)"
    )
    diag_mx = diag.add_mutually_exclusive_group()
    diag_mx.add_argument(
        "--shell",
        action="store_true",
        help="Stop after building the hollow shell and write it to the output. "
             "No seeds, no Voronoi cells, no perforation.",
    )
    diag_mx.add_argument(
        "--cutters",
        action="store_true",
        help="Build the shell and compute the Voronoi cell cutters, then write "
             "the concatenated cutters to the output (instead of the "
             "perforated shell). Useful for inspecting hole geometry.",
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Progress bars and timing info.",
    )
    p.add_argument(
        "--version",
        action="version", version=f"voronoizer {__version__}",
    )
    return p


def _validate_args(args: argparse.Namespace) -> None:
    if args.shell_thickness <= 0:
        raise SystemExit("error: --shell-thickness must be > 0")
    if args.strut_thickness <= 0:
        raise SystemExit("error: --strut-thickness must be > 0")
    if args.holes < 1:
        raise SystemExit("error: --holes must be >= 1")
    if not (0.0 < args.normal_angle < 90.0):
        raise SystemExit("error: --normal-angle must be in (0, 90)")
    if args.chamfer < 0:
        raise SystemExit("error: --chamfer must be >= 0")
    if not (0.0 < args.soft_edge_angle < 180.0):
        raise SystemExit("error: --soft-edge-angle must be in (0, 180)")
    if not args.input.exists():
        raise SystemExit(f"error: input file not found: {args.input}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    progress.set_verbose(args.verbose)

    if args.seed is None:
        args.seed = int(np.random.default_rng().integers(0, 2**31 - 1))
    print(f"voronoizer: random seed = {args.seed}", file=sys.stderr, flush=True)

    # Resolve {seed} macro in the output path so reproducible runs can keep
    # their seed visible in the filename (e.g. out_{seed}.stl -> out_42.stl).
    output_path = Path(str(args.output).replace("{seed}", str(args.seed)))

    # Imported lazily so --help / --version don't pay the trimesh import cost.
    from voronoizer.pipeline import run

    try:
        run(
            input_path=args.input,
            output_path=output_path,
            shell_thickness=args.shell_thickness,
            holes=args.holes,
            strut_thickness=args.strut_thickness,
            top_bottom_only=args.top_bottom_only,
            normal_angle_deg=args.normal_angle,
            seed=args.seed,
            repair=args.repair,
            edge_margin=args.edge_margin,
            chamfer=args.chamfer,
            soft_edge_angle_deg=args.soft_edge_angle,
            shell_only=args.shell,
            cutters_only=args.cutters,
        )
    except Exception as e:
        if args.verbose:
            raise
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
