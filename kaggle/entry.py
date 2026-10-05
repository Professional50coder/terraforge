"""Kaggle kernel entry: runs the full experiment suite on a Kaggle GPU.

Steps: clone repo -> build dataset + cache -> MAE pretrain -> train CNN / ViT / MAE-ViT over
3 seeds -> DDP check -> collect results. Large intermediates live in /kaggle/temp (not saved
as output); only small result files are copied to /kaggle/working/results.
"""
import json
import os
import shutil
import subprocess
import sys
import time

REPO = "https://github.com/Professional50coder/terraforge.git"
WORK = "/kaggle/working/terraforge"
TMP = "/kaggle/temp"
RAW, PROC = f"{TMP}/raw", f"{TMP}/processed"
RESULTS = "/kaggle/working/results"
T0 = time.time()
failures = []


def sh(cmd, critical=True):
    print(f"\n$ {cmd}  [t+{time.time() - T0:.0f}s]", flush=True)
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        failures.append(cmd)
        print(f"!! exit {r.returncode}: {cmd}", flush=True)
        if critical:
            finish()
            sys.exit(r.returncode)
    return r.returncode


def finish():
    os.makedirs(RESULTS, exist_ok=True)
    runs = f"{WORK}/runs"
    if os.path.isdir(runs):
        for f in os.listdir(runs):
            if f.endswith((".json", ".pt", ".log")):
                shutil.copy(f"{runs}/{f}", RESULTS)
    meta = {"elapsed_s": time.time() - T0, "failures": failures}
    try:
        import torch
        meta.update(torch=torch.__version__, cuda=torch.cuda.is_available(),
                    gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    except Exception as e:  # reporting must never mask the real failure
        meta["torch_error"] = repr(e)
    json.dump(meta, open(f"{RESULTS}/env.json", "w"), indent=2)
    print(json.dumps(meta, indent=2), flush=True)


os.makedirs(TMP, exist_ok=True)
sh("nvidia-smi || true", critical=False)
sh(f"git clone --depth 1 {REPO} {WORK}")
os.chdir(WORK)
os.makedirs("runs", exist_ok=True)
sh("pip install -q -e . rasterio geopandas pystac-client fastapi httpx")
sh(f"python scripts/build_dataset.py --data-dir {RAW} --out-dir {PROC}")
shutil.rmtree(RAW, ignore_errors=True)  # extracted tifs are no longer needed after caching

common = f"--processed {PROC} --workers 2 --out runs"
sh(f"python scripts/pretrain_mae.py --processed {PROC} --epochs 80 --workers 2 --out runs/mae_encoder.pt")
for seed in (0, 1, 2):
    sh(f"python scripts/train.py --model cnn --epochs 30 --seed {seed} --tag cnn_s{seed} {common}",
       critical=False)
    sh(f"python scripts/train.py --model vit --epochs 40 --lr 5e-4 --seed {seed} "
       f"--tag vit_scratch_s{seed} {common}", critical=False)
    sh(f"python scripts/train.py --model vit --epochs 30 --lr 3e-4 --seed {seed} "
       f"--pretrained runs/mae_encoder.pt --tag vit_mae_s{seed} {common}", critical=False)

sh("python -c \"from terraforge.training.distributed import run_ddp_smoke; import tempfile; "
   "print('DDP max param divergence:', run_ddp_smoke(2, tempfile.mkdtemp()))\" "
   "2>&1 | tee runs/ddp_check.log", critical=False)
finish()
