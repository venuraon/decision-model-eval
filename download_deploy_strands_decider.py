#!/usr/bin/env python3
"""Download and run Strands Decider locally on an Intel Mac.

The script deliberately uses CPU inference.  Apple Silicon-only MPS/MLX paths
are not available on the Intel Mac this project targets.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import time
import json
from urllib.request import Request, urlopen
from pathlib import Path


DEFAULT_MODEL = "StrandsAgents/strands-decider-2B-hobson-v21"
DEFAULT_PORT = 8000
MINIMUM_PYTHON = (3, 12)


def package_available() -> bool:
    return importlib.util.find_spec("strands_decider") is not None


def install_package() -> None:
    print("Installing strands-decider into the active Python environment...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "strands-decider"],
        check=True,
    )


def check_disk(path: Path, minimum_gb: float = 8.0) -> None:
    usage = shutil.disk_usage(path)
    free_gb = usage.free / (1024**3)
    if free_gb < minimum_gb:
        raise RuntimeError(
            f"Only {free_gb:.1f} GB is free at {path}; at least "
            f"{minimum_gb:.1f} GB is recommended for the model and cache."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".model-cache"),
        help="Hugging Face cache directory (relative to this script by default).",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install strands-decider if it is not already importable.",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download/validate the model, then exit without starting a server.",
    )
    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Start the server, wait for /health, then exit.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Start the server, run one example routing decision, then exit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if sys.version_info < MINIMUM_PYTHON:
        required = ".".join(map(str, MINIMUM_PYTHON))
        current = ".".join(map(str, sys.version_info[:3]))
        print(
            f"Python {required}+ is required by the documented Strands macOS "
            f"setup; this interpreter is Python {current}. Install Python "
            f"{required}+ and create the virtual environment with it.",
            file=sys.stderr,
        )
        return 2
    script_dir = Path(__file__).resolve().parent
    cache_dir = args.cache_dir.expanduser()
    if not cache_dir.is_absolute():
        cache_dir = script_dir / cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    check_disk(cache_dir)

    if not package_available():
        if not args.install:
            print(
                "strands-decider is not installed. Create/activate a virtual "
                "environment, then rerun with --install.",
                file=sys.stderr,
            )
            return 2
        install_package()

    # The CLI downloads the Hub checkpoint on first use and keeps it in the
    # configured Hugging Face cache.  Importing the package here gives a useful
    # failure before a subprocess is started, without loading 2B weights twice.
    import strands_decider  # noqa: F401

    env = os.environ.copy()
    env["HF_HOME"] = str(cache_dir)
    env["HUGGINGFACE_HUB_CACHE"] = str(cache_dir / "hub")
    executable = shutil.which("strands-decider")
    if executable is None:
        raise RuntimeError(
            "The strands-decider executable is not on PATH. Activate the "
            "environment where it was installed and rerun this script."
        )
    command = [
        executable,
        "serve",
        args.model,
        "--device",
        "cpu",
        "--port",
        str(args.port),
    ]
    print(f"Model: {args.model}")
    print("Device: cpu (Intel Mac)")
    print(f"Endpoint: http://127.0.0.1:{args.port}/v1/systemone")
    print(f"Cache: {cache_dir}")
    if args.download_only:
        # Starting the service is the portable, documented load path.  The
        # health-check branch terminates it after the checkpoint is loaded,
        # leaving the downloaded weights in the configured cache.
        args.health_check = True
    if args.smoke_test:
        args.health_check = True

    if not args.health_check:
        return subprocess.call(command, env=env)

    process = subprocess.Popen(command, env=env)
    health_url = f"http://127.0.0.1:{args.port}/health"
    try:
        for _ in range(120):
            if process.poll() is not None:
                return process.returncode or 1
            try:
                with urlopen(health_url, timeout=2) as response:
                    if response.status == 200:
                        print(f"Health check passed: {health_url}")
                        if args.smoke_test:
                            payload = {
                                "state": (
                                    "Service/form: Phone\n"
                                    "Method: Wireless\n"
                                    "Call or message type: Text Message"
                                ),
                                "questions": {
                                    "queue": {
                                        "type": "choice",
                                        "instructions": "Which queue should own this complaint?",
                                        "criteria": {
                                            "billing_and_charges": "Charges, fees, or bills.",
                                            "fraud_or_unwanted_contact": (
                                                "Robocalls, spoofing, telemarketing, or unwanted messages."
                                            ),
                                            "human_review_or_other": "Anything else or ambiguous.",
                                        },
                                    }
                                },
                            }
                            request = Request(
                                f"http://127.0.0.1:{args.port}/v1/systemone",
                                data=json.dumps(payload).encode("utf-8"),
                                headers={"Content-Type": "application/json"},
                                method="POST",
                            )
                            with urlopen(request, timeout=120) as test_response:
                                result = json.loads(test_response.read().decode("utf-8"))
                            answer = result.get("answers", {}).get("queue", {})
                            print("Smoke-test response:")
                            print(json.dumps(answer, indent=2))
                        return 0
            except Exception:
                time.sleep(1)
        print(f"Timed out waiting for {health_url}", file=sys.stderr)
        return 1
    finally:
        process.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
