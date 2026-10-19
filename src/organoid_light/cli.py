"""Command-line entry point; heavy TensorFlow import is deferred until needed."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .models import catalog, ensure_models, model_spec


def _models_command(args: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="organoid-light models")
    parser.add_argument("action", choices=["list", "download"])
    parser.add_argument("--organelle", choices=["mito", "ld", "nuc", "all"], default="all")
    parser.add_argument("--models-dir", type=Path)
    options = parser.parse_args(args)
    organs = list(catalog()["models"]) if options.organelle == "all" else [options.organelle]
    for organelle in organs:
        spec = model_spec(organelle)
        if options.action == "list":
            print(f"{organelle}: {spec['version']} ({len(spec['assets'])} checkpoint(s))")
        else:
            for path in ensure_models(organelle, options.models_dir):
                print(f"Verified model: {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        if args and args[0] == "models":
            return _models_command(args[1:])
        if args and args[0] == "doctor":
            from .runtime import bootstrap_cuda_loader, configure_runtime
            parser = argparse.ArgumentParser(prog="organoid-light doctor")
            parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
            options = parser.parse_args(args[1:])
            bootstrap_cuda_loader(args, options.device)
            _, details = configure_runtime(options.device)
            print(json.dumps(details, indent=2, default=str))
            return 0
        parser = argparse.ArgumentParser(
            description="Frozen Mito/LD/Nuc 2D segmentation of TIFF folders or grayscale Z stacks."
        )
        parser.add_argument("--version", action="version", version=__version__)
        parser.add_argument("--organelle", choices=["mito", "ld", "nuc"], required=True)
        parser.add_argument("--input", type=Path, required=True, help="TIFF file or folder")
        parser.add_argument("--output", type=Path, required=True, help="New output folder")
        parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
        parser.add_argument("--models-dir", type=Path, help="Override model download/cache directory")
        parser.add_argument("--resume", action="store_true", help="Verify and resume the same run")
        parser.add_argument("--recursive", action="store_true", help="Mirror nested input folders")
        parser.add_argument("--batch-size", type=int, default=4,
                            help="Sliding-window batch size; frozen reference default: 4")
        options = parser.parse_args(args)
        from .runtime import bootstrap_cuda_loader
        bootstrap_cuda_loader(args, options.device)
        from .pipeline import run_batch
        spec = model_spec(options.organelle)

        def factory():
            from .engine import Segmenter
            from .runtime import configure_runtime
            paths = ensure_models(options.organelle, options.models_dir)
            tf, details = configure_runtime(options.device)
            print(f"Using {spec['version']} on {details['device']}", flush=True)
            return Segmenter(spec, paths, tf, details["device"], options.batch_size), details

        state = run_batch(options.input, options.output, spec, factory, resume=options.resume,
                          recursive=options.recursive, batch_size=options.batch_size)
        print(f"COMPLETE: {len(state['files'])} TIFF file(s); outputs: {options.output}")
        return 0
    except KeyboardInterrupt:
        print("Interrupted. Rerun the same command with --resume.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
