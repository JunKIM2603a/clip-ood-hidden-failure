# Conda Environment Setup

> Pilot environment policy frozen on **2026-09-23**.

The experiment environment is intentionally split into three layers:

1. **Conda** — isolates Python and native build tools.
2. **pip / official PyTorch wheel** — installs the CUDA-enabled PyTorch runtime.
3. **Pinned third-party research code** — OpenOOD-VLM and OpenAI CLIP are fixed to exact Git commits.

This avoids depending on the old OpenOOD-VLM `environment.yml`, which still describes a Python 3.8 / PyTorch 1.8.1 / CUDA 10.1 stack that predates RTX 4090 support.

## Frozen pilot baseline

| Component | Pilot value |
| --- | --- |
| Conda env | `clip-ood` |
| Python | 3.10 |
| PyTorch | 2.9.1 |
| torchvision | 0.24.1 |
| PyTorch CUDA wheel | CUDA 12.6 (`cu126`) |
| NumPy | 1.26.4 |
| OpenOOD-VLM | commit `48b023365b68b54c87f84ab508de8a3222335985` |
| OpenAI CLIP | commit `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6` |

### Why Python 3.10

The project needs modern PyTorch for RTX 4090 while OpenOOD still eagerly imports several older packages. Python 3.10 is a conservative middle point: modern enough for current PyTorch but closer to the ecosystem against which these older OOD utilities were developed.

### Why NumPy 1.26.4 instead of NumPy 2.x

OpenOOD imports `imgaug` through its preprocessor package even when MCM/NegLabel do not use DRAEM. The released `imgaug 0.4.0` still uses NumPy APIs removed in NumPy 2.x.

Therefore NumPy 1.26.4 is an intentional compatibility pin, not an accidental old dependency.

### Why faiss-cpu although the experiment has GPUs

MCM/NegLabel do not need Faiss. However OpenOOD imports KNN-related postprocessors eagerly, and those modules import `faiss`. Installing the CPU wheel satisfies that import without introducing a second CUDA-specific binary stack.

The experiment GPU work still runs through PyTorch on the RTX 4090s.

---

# 1. Host prerequisites

Recommended primary platform:

- Linux x86_64
- NVIDIA RTX 4090
- NVIDIA driver visible through `nvidia-smi`
- Conda / Miniforge / Miniconda
- Git

Check the driver first:

```bash
nvidia-smi
```

You do **not** need to install a matching system CUDA Toolkit just to run the PyTorch wheel. The PyTorch CUDA wheel carries its own CUDA runtime. The host NVIDIA driver must be new enough to run it.

The pilot defaults to the official PyTorch CUDA 12.6 wheel. For a clean CUDA 12.6 setup, a current NVIDIA driver is recommended.

## If Conda is not installed

Miniforge is recommended because this project uses `conda-forge`.

After installing it, reopen the shell and verify:

```bash
conda --version
```

---

# 2. One-command setup

From the repository root:

```bash
git pull
bash scripts/env/setup_conda.sh
```

The script performs:

```text
environment.yml
      │
      ▼
create/update conda env: clip-ood
      │
      ▼
install PyTorch 2.9.1 + cu126
      │
      ▼
install pinned Python dependencies
      │
      ├─ NumPy 1.26.4
      ├─ scipy / sklearn / pandas
      ├─ imgaug import compatibility
      ├─ faiss-cpu
      ├─ libmr build
      ├─ Hugging Face / gdown
      └─ test utilities
      │
      ▼
install pinned OpenAI CLIP commit
      │
      ▼
clone OpenOOD-VLM into third_party/
checkout exact commit
      │
      ▼
pip install -e --no-deps
      │
      ▼
GPU + import smoke test
```

OpenOOD's own dependency resolver is deliberately bypassed with `--no-deps`. Its `setup.py` still contains broad/legacy dependencies such as `faiss-gpu`; our pilot installs the exact runtime it needs first.

---

# 3. Activate the environment later

Every new shell:

```bash
conda activate clip-ood
```

Verify manually:

```bash
python scripts/env/verify_environment.py
```

On a CPU-only PC used only for metadata or dataset preparation:

```bash
python scripts/env/verify_environment.py --allow-cpu
```

The GPU experiment PC should not use `--allow-cpu`.

---

# 4. What the verifier checks

The smoke test verifies:

- Python 3.10
- exact PyTorch / torchvision versions
- PyTorch CUDA runtime
- NumPy compatibility pin
- CUDA visibility
- matrix multiplication on every visible GPU
- GPU name / VRAM / compute capability
- imports that commonly fail in OpenOOD:
  - `imgaug`
  - `faiss`
  - `libmr`
  - OpenAI `clip`
  - `openood.preprocessors`
  - `openood.postprocessors`
  - `openood.networks`
- bundled CLIP ViT-B/32 and ViT-B/16 availability
- exact OpenOOD-VLM Git commit
- that Python imports OpenOOD from our pinned editable checkout

For the main workstation with 2×RTX 4090, the expected GPU section is conceptually:

```text
CUDA available: True
CUDA device count: 2
gpu[0]: NVIDIA GeForce RTX 4090; VRAM≈24 GiB; compute=8.9
gpu[1]: NVIDIA GeForce RTX 4090; VRAM≈24 GiB; compute=8.9
[OK] CUDA matmul smoke test
```

A machine with one compatible GPU can still reproduce a single experiment condition; it simply cannot run the planned GPU-level parallelism.

---

# 5. Why the system CUDA version may look different

It is normal to see something like:

```text
nvidia-smi: CUDA Version 12.x
torch.version.cuda: 12.6
```

Those fields describe different things:

- `nvidia-smi` reports what the installed driver can support;
- `torch.version.cuda` reports the CUDA runtime against which the PyTorch wheel was built.

The experiment should key on the **PyTorch wheel + working NVIDIA driver**, not on having an exact `nvcc` version installed globally.

---

# 6. Environment fingerprint across PCs

After the first experiment PC passes:

```bash
python scripts/env/verify_environment.py
```

freeze the critical environment:

```bash
python scripts/env/environment_lock.py freeze
```

This writes:

```text
configs/environment/pilot_environment_lock.json
```

Review and commit it:

```bash
git add configs/environment/pilot_environment_lock.json
git commit -m "Freeze pilot experiment environment"
git push
```

The lock stores the versions of critical scientific packages plus:

- Python version
- PyTorch CUDA runtime
- OpenOOD-VLM commit
- OpenAI CLIP commit

It deliberately does **not** require every PC to have the exact same NVIDIA driver version. Different sufficiently recent drivers are acceptable because the PyTorch CUDA runtime is fixed.

On another PC:

```bash
git pull
bash scripts/env/setup_conda.sh
conda activate clip-ood
python scripts/env/environment_lock.py check
```

This gives the desired workflow:

```text
                 Git repository
              environment specs
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
        PC A       PC B       PC C
          │          │          │
     setup_conda setup_conda setup_conda
          │          │          │
          ▼          ▼          ▼
       lock OK     lock OK     lock OK
```

---

# 7. Dataset installation comes after environment verification

Once:

```bash
python scripts/env/verify_environment.py
```

passes, install data:

```bash
python scripts/data/setup_datasets.py --datasets ood
```

For ImageNet through the gated Hugging Face mirror:

```bash
hf auth login

python scripts/data/setup_datasets.py \
  --datasets all \
  --imagenet-source hf
```

See `docs/dataset_setup.md` for details.

---

# 8. Re-creating vs updating the environment

The setup script is idempotent enough for ordinary Git updates:

```bash
git pull
bash scripts/env/setup_conda.sh
```

If the environment becomes badly inconsistent, recreate it rather than manually repairing dozens of packages:

```bash
conda deactivate || true
conda env remove -n clip-ood

bash scripts/env/setup_conda.sh
```

This is preferable to repeatedly running arbitrary `pip install -U ...`, because uncontrolled upgrades make baseline reproduction harder.

---

# 9. Do not install OpenOOD with its dependencies again

After setup, avoid:

```bash
pip install -e third_party/OpenOOD-VLM
```

without `--no-deps`.

That command can let OpenOOD's broad legacy dependency declarations modify the frozen environment.

The setup script already installs it correctly:

```bash
pip install --no-deps -e third_party/OpenOOD-VLM
```

---

# 10. GPU allocation for the later experiment

With two RTX 4090s, a natural pilot layout is:

```text
GPU 0
├─ MCM / ViT-B/32
└─ prompt conditions

GPU 1
├─ NegLabel / ViT-B/32
└─ prompt conditions
```

or, after baseline reproduction:

```text
GPU 0 → backbone / OOD condition A
GPU 1 → backbone / OOD condition B
```

Use `CUDA_VISIBLE_DEVICES` to isolate runs:

```bash
CUDA_VISIBLE_DEVICES=0 python ...
CUDA_VISIBLE_DEVICES=1 python ...
```

The Conda environment is shared; only the visible device changes.

---

# 11. Files that define the environment

```text
environment.yml
requirements/
└── experiment.txt

configs/environment/
├── versions.env
└── pilot_environment_lock.json

scripts/env/
├── setup_conda.sh
├── verify_environment.py
└── environment_lock.py
```

The distinction is:

- `environment.yml`: Python + native build tools
- `requirements/experiment.txt`: Python library pins
- `versions.env`: PyTorch CUDA and external Git commits
- `pilot_environment_lock.json`: observed first-PC environment fingerprint
- `setup_conda.sh`: reconstruct everything
- `verify_environment.py`: test it
- `environment_lock.py`: compare experiment PCs
