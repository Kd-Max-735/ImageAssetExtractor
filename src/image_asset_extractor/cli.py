from __future__ import annotations

import argparse
import json
import sys

from .errors import ExtractorError
from .schemas.config import ExtractionConfig
from .services.extraction import ExtractionEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="image-asset-extractor")
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract = subparsers.add_parser("extract", help="extract transparent PNG assets")
    extract.add_argument("input")
    extract.add_argument("--output", required=True)
    extract.add_argument("--background-type", default="auto")
    extract.add_argument("--asset-mode", default="icon")
    extract.add_argument("--shadow-mode", default="auto")
    extract.add_argument("--text-mode", default="auto")
    extract.add_argument("--min-area-ratio", type=float, default=0.0001)
    extract.add_argument("--merge-distance-ratio", type=float, default=0.01)
    extract.add_argument("--padding-ratio", type=float, default=0.05)
    extract.add_argument("--background-tolerance", type=float, default=15)
    extract.add_argument("--output-prefix", default="icon")
    extract.add_argument("--number-digits", type=int, default=3)
    extract.add_argument("--output-canvas", choices=("tight", "square"), default="tight")
    extract.add_argument("--output-format", choices=("png",), default="png")
    extract.add_argument("--canvas-size", type=int)
    extract.add_argument("--edge-feather", type=float, default=0)
    extract.add_argument("--split-strength", type=float, default=0.5)
    extract.add_argument("--include-zip", action=argparse.BooleanOptionalAction, default=True)
    extract.add_argument("--use-rmbg", action="store_true")
    extract.add_argument("--use-sam31", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ExtractionConfig(
        background_type=args.background_type, asset_mode=args.asset_mode,
        shadow_mode=args.shadow_mode, text_mode=args.text_mode, min_area_ratio=args.min_area_ratio,
        merge_distance_ratio=args.merge_distance_ratio, padding_ratio=args.padding_ratio,
        background_tolerance=args.background_tolerance, output_prefix=args.output_prefix,
        number_digits=args.number_digits, output_canvas=args.output_canvas, output_format=args.output_format, canvas_size=args.canvas_size,
        edge_feather=args.edge_feather, split_strength=args.split_strength, include_zip=args.include_zip,
        use_rmbg=args.use_rmbg, use_sam31=args.use_sam31,
    )
    try:
        result = ExtractionEngine().extract(args.input, args.output, config)
    except ExtractorError as exc:
        print(json.dumps(exc.as_dict(), ensure_ascii=False), file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(json.dumps({"code": "CLI_ERROR", "message": str(exc), "details": {}}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result.metadata(), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
