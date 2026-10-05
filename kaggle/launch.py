"""Push the training kernel to Kaggle, wait for it, and download the results.

    python kaggle/launch.py push      # create/update the kernel and start it
    python kaggle/launch.py status
    python kaggle/launch.py pull      # download output into runs_kaggle/

Credentials: Kaggle's CLI reads KAGGLE_USERNAME / KAGGLE_KEY from the environment. They are
taken from the .env named by TERRAFORGE_ENV_FILE with an allowlist (only those two keys), passed
to the child process only, and never printed or written to disk. The kernel is private.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from terraforge.multimodal.explainer import load_env

HERE = Path(__file__).parent
SLUG = "terraforge-train"


def kaggle_env() -> dict:
    env = dict(os.environ)
    path = os.environ.get("TERRAFORGE_ENV_FILE")
    if path:
        snapshot = dict(os.environ)
        load_env(path, allow=("KAGGLE_USERNAME", "KAGGLE_KEY"))
        env = dict(os.environ)
        os.environ.clear(); os.environ.update(snapshot)  # keep this process clean
    if not (env.get("KAGGLE_USERNAME") and env.get("KAGGLE_KEY")):
        raise SystemExit("KAGGLE_USERNAME / KAGGLE_KEY not available; set TERRAFORGE_ENV_FILE")
    return env


def run(args, env):
    r = subprocess.run([sys.executable, "-m", "kaggle.cli", *args], env=env,
                       capture_output=True, text=True)
    print((r.stdout + r.stderr).strip())
    return r.returncode


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    env = kaggle_env()
    user = env["KAGGLE_USERNAME"]
    ref = f"{user}/{SLUG}"
    if cmd == "push":
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "entry.py").write_text((HERE / "entry.py").read_text())
            (Path(d) / "kernel-metadata.json").write_text(json.dumps({
                "id": ref, "title": SLUG, "code_file": "entry.py", "language": "python",
                "kernel_type": "script", "is_private": "true", "enable_gpu": "true",
                "enable_internet": "true", "dataset_sources": [], "competition_sources": [],
                "kernel_sources": []}))
            sys.exit(run(["kernels", "push", "-p", d], env))
    elif cmd == "status":
        sys.exit(run(["kernels", "status", ref], env))
    elif cmd == "pull":
        out = Path("runs_kaggle"); out.mkdir(exist_ok=True)
        sys.exit(run(["kernels", "output", ref, "-p", str(out)], env))
    else:
        raise SystemExit("usage: launch.py push|status|pull")


if __name__ == "__main__":
    main()
