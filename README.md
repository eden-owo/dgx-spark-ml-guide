# DGX Spark PyTorch & CUDA Guide: Running ML on GB10 Blackwell

> **The definitive guide to running PyTorch and ML workloads on NVIDIA DGX Spark with GB10 GPU.**
> Written by someone who spent 72+ hours debugging so you don't have to.

---

## Document Information

| | |
|---------------------|----------------------------------------------------------------------|
| **Author** | **Martim Ramos** —  DevOps Lead @ AXA (The Lead tha codes 90%+ of the time.)  |
| **Hardware** | NVIDIA DGX Spark with GB10 (Blackwell architecture, sm_121) |
| **Last Updated** | April 2026 |
| **Verified Working** | Video generation, lip-sync avatars, image diffusion, music LoRA training, **Chatterbox TTS (multilingual zero-shot voice cloning, incl. PT-PT)** |
| **License** | MIT |

---

## Why This Guide Exists

**I bought a DGX Spark expecting to run video generation models within hours. Instead, I spent three days fighting errors that had zero documentation online.**

The DGX Spark is NVIDIA's first desktop AI supercomputer featuring the GB10 Blackwell chip. It's powerful — 128GB unified memory, 1 PFLOP of AI performance — but almost nothing works out of the box:

- ❌ Standard PyTorch doesn't support it
- ❌ NGC containers don't support it  
- ❌ Flash Attention has no prebuilt wheels
- ❌ Most ARM64 + CUDA packages don't exist
- ❌ TorchCodec installs a dummy package
- ❌ TensorBoard's Docker image is x86_64 only

This guide documents every problem I encountered and how I solved it. If you're Googling a DGX Spark error at 2am, this is for you.

---

## Quick Start: What You Need to Know in 60 Seconds

**The core issue:** GB10 uses Blackwell architecture (sm_121), which requires PyTorch nightly builds. The sm_120 target in PyTorch nightly is binary compatible with sm_121.

**The solution for 90% of projects:**

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

**If you're using MMLab (mmcv, mmdet, mmpose):** You need Docker with Python 3.10. See [Challenge 4](#challenge-4-python-312-breaks-mmlab-stack).

**If builds fail with CUDA errors:** Your system has CUDA 13.0 but PyTorch uses 12.8. See [Challenge 2](#challenge-2-cuda-version-mismatch-when-compiling-extensions).

**If your model fails at runtime with `nvrtc: error: invalid value for --gpu-architecture`** (typical for TTS / audio / FFT pipelines): the bundled NVRTC doesn't know sm_121. See [Challenge 14](#challenge-14-runtime-nvrtc-error-bundled-libnvrtcso12-doesnt-know-sm_121).

**If a project's `pyproject.toml` pins `torch==2.6.0`** (Chatterbox-TTS, older XTTS, many HF repos): a normal install will silently clobber your cu128 nightly. Use `--no-deps`. See [Challenge 15](#challenge-15-project-pins-torch260-and-clobbers-your-cu128-nightly).

---

## System Specifications

| Component | Specification |
|-----------|---------------|
| **GPU** | NVIDIA GB10 (Blackwell architecture, compute capability sm_121) |
| **Platform** | ARM64 (aarch64) — not x86_64 |
| **System CUDA** | 13.0 |
| **Unified Memory** | 128GB shared between CPU and GPU |
| **OS** | Ubuntu 24.04 LTS |
| **Driver** | 580.95.05 |

**Key insight:** The 128GB is unified memory — CPU and GPU share the same pool. This means CPU offloading provides zero benefit (unlike discrete GPUs), but you can load massive models without OOM errors.

---

## Challenge 1: PyTorch Doesn't Detect the GPU

### What error will I see?

```python
>>> import torch
>>> torch.cuda.is_available()
False
```

Or errors mentioning "unsupported compute capability" or "sm_121".

### Why does this happen?

Standard PyTorch releases (2.5.x and earlier) only support up to sm_90 (Hopper architecture). The GB10 uses sm_121 (Blackwell), which is only supported in PyTorch nightly builds with CUDA 12.8.

### How do I fix it?

Install PyTorch nightly with CUDA 12.8:

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

The sm_120 architecture in these builds is binary compatible with sm_121.

### How do I verify it's working?

```bash
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')"
```

**Expected output:**

```
PyTorch: 2.11.0.dev2025XXXX+cu128
CUDA: True
Device: NVIDIA GB10
```

---

## Challenge 2: CUDA Version Mismatch When Compiling Extensions

### What error will I see?

```
ModuleNotFoundError: No module named 'mmcv._ext'
```

Or during compilation:

```
RuntimeError: The detected CUDA version (13.0) mismatches the version that was used to compile PyTorch (12.8)
```

### Why does this happen?

Your DGX Spark has CUDA 13.0 installed system-wide, but PyTorch nightly bundles CUDA 12.8. When compiling CUDA extensions (mmcv, flash_attn, custom operators), the compiler uses CUDA 13.0 headers while linking against PyTorch's CUDA 12.8 runtime. This version conflict causes silent failures.

### How do I fix it?

**Option A: Use Docker (Recommended)**

Docker isolates the CUDA environment completely:

```dockerfile
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="12.0"

RUN pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

**Option B: Patch PyTorch's cpp_extension.py**

Edit `.venv/lib/python3.12/site-packages/torch/utils/cpp_extension.py` around line 546 to allow ±1 CUDA major version mismatch. This is a workaround — Docker is cleaner.

---

## Challenge 3: No ARM64 Wheels for Common Packages

### Which packages are affected?

| Package | ARM64 + CUDA Wheel? | Solution |
|---------|---------------------|----------|
| mmcv | ❌ No | Build from source |
| flash_attn | ❌ No | Build from source |
| decord | ❌ No | Feature unavailable |
| xtcocotools | ❌ No | Build from source |
| spacy (pinned versions) | ⚠️ Some versions missing | Use `>=3.7.0` |

### How do I build mmcv from source?

```bash
MMCV_WITH_OPS=1 FORCE_CUDA=1 \
pip install git+https://github.com/open-mmlab/mmcv.git@v2.1.0 --no-build-isolation
```

**Build time:** Approximately 25 minutes (compiling CUDA kernels).

### How do I build Flash Attention 2 from source?

```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git submodule update --init csrc/cutlass

CUDA_HOME=/usr/local/cuda \
TORCH_CUDA_ARCH_LIST="12.0" \
FLASH_ATTN_CUDA_ARCHS="120" \
FLASH_ATTENTION_FORCE_BUILD=TRUE \
MAX_JOBS=4 \
pip install -e . --no-build-isolation
```

**Build time:** 15-20 minutes.

### What about decord?

There's no ARM64 wheel and no straightforward build path. Features requiring decord (like some video-to-video pipelines) must be disabled or refactored to use alternative libraries.

### How do I handle pinned spacy versions?

If a project pins `spacy==3.8.4` (or similar) and no ARM64 wheel exists:

```dockerfile
RUN sed -i 's/spacy==3.8.4/spacy>=3.7.0/' /app/requirements.txt
```

---

## Challenge 4: Python 3.12 Breaks MMLab Stack

### What error will I see?

```
AttributeError: module 'pkgutil' has no attribute 'ImpImporter'
```

### Why does this happen?

Ubuntu 24.04 ships with Python 3.12 by default. The MMLab ecosystem (mmcv, mmdet, mmpose) depends on `pkg_resources`, which uses deprecated Python APIs that were removed in Python 3.12.

### How do I fix it?

Use Python 3.10 for MMLab projects:

```dockerfile
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y python3.10 python3.10-venv python3-pip
RUN ln -sf /usr/bin/python3.10 /usr/bin/python
```

**Note:** Non-MMLab projects work fine with Python 3.12.

---

## Challenge 5: PyTorch 2.6+ Breaks Old Checkpoints

### What error will I see?

```
_pickle.UnpicklingError: Weights only load failed... weights_only=True
```

### Why does this happen?

PyTorch 2.6+ changed `torch.load()` to use `weights_only=True` by default as a security fix (CVE-2025-32434). Old model checkpoints that contain non-tensor objects fail to load.

### How do I fix it?

Patch `torch.load` at application startup:

```python
import functools
import torch

_original_torch_load = torch.load

@functools.wraps(_original_torch_load)
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load
```

Add this before any model loading code runs.

---

## Challenge 6: NGC PyTorch Containers and GB10 — Status as of April 2026

### Short version

| NGC tag | GB10 (sm_121) support | Recommended? |
|---------|------------------------|---------------|
| `nvcr.io/nvidia/pytorch:24.10-py3` and earlier | ❌ No | No |
| `nvcr.io/nvidia/pytorch:25.10-py3` and later | ✅ **Yes** | **Yes — often the cleanest path** |

### What error will I see on old NGC images?

```
WARNING: Detected NVIDIA GB10 GPU, which is not yet supported
```

### What ships in the working `pytorch:25.10-py3` image?

Verified inside the running container on GB10:

```text
torch: 2.9.0a0+145a3a7bda.nv25.10
cuda: True
device: NVIDIA GB10
cap: (12, 1)
arch_list: ['sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_110', 'sm_120', 'compute_120']
```

Container CUDA: `13.0.88` (matches host). Container NVRTC: `libnvrtc.so.13` (matches host). The `compute_120` PTX JITs to sm_121 cleanly — meaning the runtime ops that break the cu128-nightly venv path (see [Challenge 14](#challenge-14-runtime-nvrtc-error-bundled-libnvrtcso12-doesnt-know-sm_121)) **just work** here:

```python
>>> torch.randn(1024, dtype=torch.complex64, device="cuda").abs()              # OK
>>> torch.fft.rfft(torch.randn(2048, device="cuda")).abs()                     # OK
```

### When should I use NGC vs the venv-with-cu128-nightly approach?

- **Use NGC `pytorch:25.10-py3`+** when you want a single, NVIDIA-tested CUDA + cuDNN + NVRTC + torch stack and don't care about a ~10–30 GB image. **No** torch nightly. **No** NVRTC symlink. **No** `--no-deps` dance for torch-pinning projects (because NGC's torch isn't on PyPI's resolver radar).
- **Use the venv path** when you need a small image, want bleeding-edge nightly features, or are building inside an existing non-NGC base.

### How do I use NGC for ComfyUI specifically?

See [Challenge 16](#challenge-16-comfyui-on-dgx-spark-the-official-playbook-is-fragile-use-ngc).

### Old advice (kept for context)

Earlier versions of this guide recommended `nvidia/cuda:12.8.0-devel-ubuntu22.04` plus a manual PyTorch nightly install, because NGC PyTorch ≤24.10 didn't support sm_121. That guidance is now stale for new builds — use NGC ≥25.10 if you want the easiest GB10 stack.

---

## Challenge 7: Flash Attention 3 Not Available

### Can I use Flash Attention 3 on DGX Spark?

No. Flash Attention 3 (using CuTe DSL) only supports sm_90 through sm_110. The GB10's sm_121 is not supported.

### What should I use instead?

Use Flash Attention 2 built for sm_120 (binary compatible with sm_121), or fall back to PyTorch's native scaled dot-product attention:

```python
from torch.nn.functional import scaled_dot_product_attention
```

---

## Challenge 8: Understanding Unified Memory

### How is DGX Spark memory different?

Unlike discrete GPUs where CPU RAM and GPU VRAM are separate, DGX Spark uses unified memory — the 128GB is shared between CPU and GPU.

| Traditional GPU | DGX Spark |
|-----------------|-----------|
| CPU offloading saves VRAM | Offloading has no benefit (same memory pool) |
| PCIe transfer overhead | No transfer overhead |
| Discrete GPU memory is faster | Memory bandwidth is shared |

### What does this mean for my workloads?

- **Don't use** `--offload_model` or CPU offloading flags — they provide zero benefit
- Large models (50GB+) load easily without OOM errors
- Performance is compute-bound, not memory-bound
- Expect ~65 seconds/step for 5B parameter diffusion models

---

## Challenge 9: Ubuntu 24.04 Package Name Changes

### What error will I see?

```
E: Unable to locate package libgl1-mesa-glx
```

### How do I fix it?

Package names changed in Ubuntu 24.04. Either use Ubuntu 22.04 as your Docker base (recommended for compatibility) or update package names:

```dockerfile
# Ubuntu 22.04 (recommended)
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# Or for Ubuntu 24.04, use new names:
RUN apt-get install -y libgl1  # Not libgl1-mesa-glx
```

---

## Challenge 10: Docker Runtime Configuration

### What error will I see?

```
Error response from daemon: unknown or invalid runtime name: nvidia
```

### Why does this happen?

Using `runtime: nvidia` in docker-compose.yml requires the NVIDIA Container Toolkit runtime to be explicitly configured in Docker daemon settings, which may not be set up by default.

### How do I fix it?

Use the `deploy.resources.reservations` syntax instead:

```yaml
# ❌ Don't use:
runtime: nvidia

# ✅ Use instead:
services:
  myservice:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu, compute, utility]
```

### How do I verify GPU access works?

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

---

## Challenge 11: peft and transformers Version Conflicts

### What error will I see?

```
ModuleNotFoundError: No module named 'transformers.modeling_layers'
```

### Why does this happen?

Projects may pin older `transformers` versions, but the latest `peft` library requires newer transformers APIs.

### How do I fix it?

Pin peft to a version compatible with your transformers:

| transformers version | Compatible peft versions |
|----------------------|--------------------------|
| 4.50.0 | 0.10.x - 0.13.x |
| 4.51+ | 0.14.x+ |

```dockerfile
# For transformers 4.50.0:
RUN pip install "peft>=0.10.0,<0.14.0"

# Or upgrade both for latest compatibility:
RUN pip install "transformers>=4.51.0" "peft>=0.17.0"
```

---

## Challenge 12: TorchCodec Not Available

### What error will I see?

```
ImportError: TorchCodec is required for load_with_torchcodec. Please install torchcodec to use this function.
```

Or when saving audio:

```
ImportError: TorchCodec is required for save_with_torchcodec. Please install torchcodec to use this function.
```

### Why does this happen?

PyTorch nightly's torchaudio (2.10.0+) uses TorchCodec as the default backend for both loading and saving audio. However, `pip install torchcodec` installs a dummy placeholder package (version 0.0.0.dev0), not the actual library. The real TorchCodec isn't available for ARM64.

### How do I fix it?

Use the `soundfile` library instead of torchaudio for both loading and saving:

```python
import soundfile as sf
import torch

# Loading audio
# Instead of: audio, sr = torchaudio.load(filename)
audio_np, sr = sf.read(filename, dtype='float32')
audio = torch.from_numpy(audio_np)
if audio.dim() == 1:
    audio = audio.unsqueeze(0)  # mono: add channel dim
else:
    audio = audio.t()  # stereo: (samples, channels) -> (channels, samples)

# Saving audio
# Instead of: torchaudio.save(path, tensor, sr)
sf.write(path, tensor.cpu().t().numpy(), sr)  # (channels, samples) -> (samples, channels)
```

Install with: `pip install soundfile`

---

## Challenge 13: TensorFlow/TensorBoard Docker Images x86_64 Only

### What error will I see?

```
WARNING: The requested image's platform (linux/amd64) does not match the detected host platform (linux/arm64/v8)
```

### Why does this happen?

The official `tensorflow/tensorflow` Docker image is built for x86_64 only. Running it on ARM64 triggers QEMU emulation, which is extremely slow and may crash.

### How do I fix it?

Use `python:3.10-slim` (which has native ARM64 builds) and install TensorBoard at runtime:

```yaml
services:
  tensorboard:
    image: python:3.10-slim
    command: >
      bash -c "pip install -q tensorboard && tensorboard --logdir=/logs --host=0.0.0.0 --port=6006"
    volumes:
      - ./logs:/logs
    ports:
      - "6006:6006"
```

**Note:** You'll see "TensorFlow installation not found - running with reduced feature set" — this is normal and doesn't affect log viewing.

---

## Challenge 14: Runtime NVRTC Error — Bundled libnvrtc.so.12 Doesn't Know sm_121

This is the most-missed gotcha on GB10 in 2026. It does **not** trigger at install time and `torch.cuda.is_available()` returns `True`. It hits the first time your model runs an op that needs JIT codegen — common in audio / TTS / ASR / FFT pipelines.

### What error will I see?

```
nvrtc: error: invalid value for --gpu-architecture (-arch)
```

…buried at the bottom of a long traceback that includes generated CUDA source like:

```
abs_kernel<std::complex<float>>(arg0[j])
```

The traceback typically passes through `torch.fft.rfft(...).abs()`, `torchaudio.compliance.kaldi.fbank`, `torchaudio.transforms.MelSpectrogram`, or any custom GPU pipeline that does `complex.abs()`.

### Why does this happen?

PyTorch nightly `cu128` ships its own NVRTC at:

```
<venv>/lib/python3.X/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12
```

That bundled NVRTC is from CUDA 12.8 — released before Blackwell GB10 — and **does not recognize sm_121** as a valid `--gpu-architecture` value.

`torch.cuda.get_arch_list()` confirms it:

```python
>>> import torch
>>> torch.cuda.get_device_capability(0)
(12, 1)
>>> torch.cuda.get_arch_list()
['sm_80', 'sm_90', 'sm_100', 'sm_120']   # sm_121 missing
```

Pre-compiled ATen kernels for `sm_120` run on GB10 (forward compatible). But the moment PyTorch's **jiterator** falls back to NVRTC for an uncommon type combo (notably `abs(complex64)` returned by FFT ops), it asks NVRTC to generate **sm_121** code — and the bundled 12.8 lib errors out.

### How do I fix it?

Replace the bundled NVRTC 12 with a symlink to the system NVRTC 13 (CUDA 13.0 ships one that knows sm_121). The C ABI is stable enough across this version gap that PyTorch's `dlopen` + `dlsym` still resolves what it needs:

```bash
cd <venv>/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib
mv libnvrtc.so.12 libnvrtc.so.12.orig
ln -s /usr/local/cuda-13.0/lib64/libnvrtc.so.13 libnvrtc.so.12
```

For Docker images, do the same inside the container's site-packages.

### How do I verify the fix?

```bash
python3 -c "
import torch, torchaudio.compliance.kaldi as Kaldi
au = torch.randn(16000).cuda()
out = Kaldi.fbank(au.unsqueeze(0), num_mel_bins=80)
print('fbank OK on', out.device, 'shape', tuple(out.shape))
"
```

Without the fix: NVRTC arch error.
With the fix: `fbank OK on cuda:0 shape (98, 80)`.

### Why not patch the jiterator?

Jiterator is C++ side; the device cap is read from `at::cuda::getCurrentDeviceProperties()->major/minor`. Overriding it requires patching `libtorch_cuda.so`. The NVRTC symlink is the smallest change that unblocks the widest set of ops.

### Will this be needed forever?

No — once PyTorch nightly bumps its bundled NVRTC to 13.x (which natively knows sm_121), this becomes unnecessary. As of April 2026, the symlink is still required.

---

## Challenge 15: Project Pins `torch==2.6.0` and Clobbers Your cu128 Nightly

### Which projects hit this?

Many TTS, ASR, and diffusion repos pin specific torch versions in `pyproject.toml` or `requirements.txt`. Common offenders:

- `chatterbox-tts` (pins `torch==2.6.0`, `torchaudio==2.6.0`, `numpy<2.0.0`)
- Older XTTS / Coqui-TTS forks
- Many HuggingFace example repos and demo apps

### What happens if I install normally?

`pip install <project>` will resolve the pinned `torch==2.6.0`, find no `aarch64 + cu128` wheel for it, and silently fall back to `torch==2.6.0+cpu` (or fail outright). You lose GB10 acceleration without an obvious error.

### How do I install without breaking torch?

Install the package itself with `--no-deps`, then re-install its dependencies individually (omitting the torch/torchaudio/numpy pins):

```bash
# 1. Project itself, no deps
pip install -e ./chatterbox --no-deps

# 2. Reinstall the other deps (skip torch/torchaudio/numpy pins)
pip install librosa s3tokenizer transformers diffusers conformer safetensors \
            spacy-pkuseg pykakasi pyloudnorm omegaconf \
            'resemble-perth @ git+https://github.com/resemble-ai/Perth.git@master'

# 3. Pin onnx==1.16.0 if onnx is in the tree
#    (pre-built aarch64 wheel exists; later versions try to build from source)
pip install onnx==1.16.0
```

`pip` will print `dependency resolver does not currently take into account...` warnings about torch/numpy version mismatches. **These are expected and harmless** — pip's resolver doesn't model `aarch64 + cu128` reality.

### How do I verify torch is still on cu128?

```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: 2.12.0.dev...+cu128 True NVIDIA GB10
```

If it prints `+cpu` or a `2.6.0` version, the install clobbered torch — re-do step 1 with `--no-deps`.

---

## Challenge 16: ComfyUI on DGX Spark — The Official Playbook Is Fragile, Use NGC

### What's the official playbook and why does it break?

NVIDIA publishes a [DGX Spark ComfyUI playbook](https://github.com/nvidia/dgx-spark-playbooks/tree/main/nvidia/comfy-ui) that walks you through a **host-venv** install:

```bash
python3 -m venv comfyui-env
source comfyui-env/bin/activate

# Step 3: install torch
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# Step 5 (after cloning ComfyUI):
pip install -r requirements.txt
```

The playbook is **fragile** for three reasons:

1. **Step 5 can silently overwrite Step 3.** If ComfyUI's `requirements.txt` (or any custom node's `requirements.txt`) pins a different `torch` / `torchvision` / `torchaudio`, `pip install -r requirements.txt` resolves to that pin and replaces the GB10-capable wheels you just installed. ComfyUI mainline currently does not pin torch — but **most popular custom nodes do**, and the moment you install one, your torch can be downgraded.
2. **Nothing handles the bundled-NVRTC sm_121 issue** — see [Challenge 14](#challenge-14-runtime-nvrtc-error-bundled-libnvrtcso12-doesnt-know-sm_121). PyPI torch wheels (cu128 *and* cu130) ship NVRTC older than CUDA 13's, so any custom node that runs FFT-based ops on GPU can crash at runtime even after a "successful" install.
3. **Memory advice is too coarse** — `echo 3 > /proc/sys/vm/drop_caches` helps in some cases, but real GB10 OOMs come from custom nodes accumulating model state. The playbook doesn't mention `--lowvram`, `--cpu-vae`, or unified-memory implications (see Challenge 8).

### What error will I see when the playbook breaks?

After installing a custom node, ComfyUI starts but runs on CPU:

```
Set vram state to: DISABLED
Device: cpu
```

Or, on the first GPU op:

```
torch.cuda.is_available() = False
```

Or, if torch is still GPU-capable but a custom node uses FFT/complex ops:

```
nvrtc: error: invalid value for --gpu-architecture (-arch)
```

(That last one is Challenge 14, hit through a different door.)

Diagnostic one-liner:

```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

If you see `+cpu` or `False`, your torch was clobbered.

### How do I fix it? Containerize on NGC.

Use this Dockerfile — verified working on GB10 in production:

```dockerfile
FROM nvcr.io/nvidia/pytorch:25.10-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN useradd -m -u 1100 comfy
WORKDIR /home/comfy

RUN git clone https://github.com/comfyanonymous/ComfyUI.git /home/comfy/ComfyUI
WORKDIR /home/comfy/ComfyUI

# CRITICAL: Strip torch / torchvision / torchaudio from requirements.txt
# BEFORE pip install, so NGC's GB10-capable torch is preserved.
# The regex is precise: it does NOT match torchsde (needed by samplers).
RUN sed -i '/^torch[>=< ]/d; /^torch$/d; /^torchvision/d; /^torchaudio/d' requirements.txt \
    && pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /home/comfy/ComfyUI/models/checkpoints \
              /home/comfy/ComfyUI/output \
    && chown -R comfy:comfy /home/comfy

USER comfy
EXPOSE 8188
CMD ["python3", "main.py", "--listen", "0.0.0.0", "--port", "8188"]
```

### Why the sed regex matters (don't simplify it)

ComfyUI samplers depend on `torchsde`. The naive simplification `grep -v torch` (or `sed '/torch/d'`) **removes torchsde too** and silently breaks several samplers (DPM++ SDE, DPM2 a, etc.) at workflow runtime — long after install. The four-pattern regex strips exactly:

- `^torch[>=< ]` — `torch>=2.0`, `torch <3`, `torch ==2.6.0`, etc.
- `^torch$` — the bare `torch`
- `^torchvision` — any `torchvision*` line
- `^torchaudio` — any `torchaudio*` line

…and leaves `torchsde`, `torchmetrics`, `torch-something-else` alone. Even on current ComfyUI (which doesn't pin torch in mainline `requirements.txt`), keep the sed line as defense — custom node `requirements.txt` files are a moving target.

### docker-compose.yml snippet

```yaml
services:
  comfyui:
    build: ./back/comfyui
    container_name: bard-comfyui
    ports:
      - "8188:8188"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    volumes:
      - /opt/ai-models/comfyui:/home/comfy/ComfyUI/models
      - ./output:/home/comfy/ComfyUI/output
```

(`runtime: nvidia` does not work — see [Challenge 10](#challenge-10-docker-runtime-configuration).)

### How do I verify the container is actually using the GPU?

```bash
docker exec -it bard-comfyui python3 -c "
import torch
print('torch:', torch.__version__)
print('cuda:', torch.cuda.is_available())
print('device:', torch.cuda.get_device_name(0))
print('cap:', torch.cuda.get_device_capability(0))
"
```

Expected output:

```
torch: 2.9.0a0+145a3a7bda.nv25.10
cuda: True
device: NVIDIA GB10
cap: (12, 1)
```

### What about container size?

The image is ~30 GB (NGC base is large). If that's a problem, use the venv + cu128 nightly + NVRTC symlink approach — but accept that custom nodes will more frequently break your install.

### What about exit code 137?

That's the kernel OOM killer, not an install issue. GB10's 128 GB unified memory is shared with the host — large workflows + a heavy host workload can hit the cap. Mitigations: launch ComfyUI with `--lowvram` or `--normalvram`, avoid loading multiple checkpoints simultaneously, and see [Challenge 8](#challenge-8-understanding-unified-memory) for unified-memory specifics.

---

## Quick Reference: Error → Solution Table

| Error Message | Cause | Solution |
|---------------|-------|----------|
| `torch.cuda.is_available()` returns False | PyTorch doesn't support sm_121 | Install PyTorch nightly with CUDA 12.8 |
| `No module named 'mmcv._ext'` | CUDA version mismatch | Use Docker with CUDA 12.8 base |
| `weights_only=True` pickle error | PyTorch 2.6+ security change | Patch `torch.load` at startup |
| `GB10 GPU not yet supported` | NGC container limitation | Use base CUDA image, not NGC |
| `ImpImporter` AttributeError | Python 3.12 + old pkg_resources | Use Python 3.10 |
| `sm_121 not supported` | Flash Attention prebuilt binaries | Build Flash Attention 2 from source |
| `CUDA version mismatch` | System CUDA 13.0 vs PyTorch 12.8 | Use Docker or patch cpp_extension.py |
| `decord` import error | No ARM64 wheel exists | Disable features requiring decord |
| `unknown runtime name: nvidia` | Docker runtime not configured | Use `deploy.resources.reservations` |
| `No module named 'transformers.modeling_layers'` | peft/transformers mismatch | Pin peft to compatible version |
| `No matching distribution for spacy==X.X.X` | Pinned spacy lacks ARM64 wheel | Use `spacy>=3.7.0` |
| `TorchCodec is required` | torchaudio nightly needs TorchCodec | Use `soundfile` for audio I/O |
| `nvrtc: error: invalid value for --gpu-architecture` (runtime) | Bundled NVRTC 12.8 doesn't know sm_121 | Symlink system NVRTC 13 — see [Challenge 14](#challenge-14-runtime-nvrtc-error-bundled-libnvrtcso12-doesnt-know-sm_121) |
| Silent torch downgrade to `2.6.0+cpu` after `pip install` | Project pins `torch==2.6.0` | Install with `--no-deps` — see [Challenge 15](#challenge-15-project-pins-torch260-and-clobbers-your-cu128-nightly) |
| `abs_kernel<std::complex<float>>` in CUDA codegen traceback | Jiterator NVRTC codegen for sm_121 | Same fix as Challenge 14 |
| ComfyUI: `Set vram state to: DISABLED` / `Device: cpu` after install | Custom node's `requirements.txt` clobbered torch | Use NGC `pytorch:25.10-py3` + sed-strip torch/torchvision/torchaudio — see [Challenge 16](#challenge-16-comfyui-on-dgx-spark-the-official-playbook-is-fragile-use-ngc) |
| Samplers (DPM++ SDE, DPM2 a) fail at workflow runtime | `torchsde` accidentally stripped by `grep -v torch` | Use precise regex: `^torch[>=< ]`, `^torch$`, `^torchvision`, `^torchaudio` only |
| `WARNING: Detected NVIDIA GB10 GPU, which is not yet supported` (NGC) | Old NGC image (≤24.10) | Upgrade to `nvcr.io/nvidia/pytorch:25.10-py3` or later — see [Challenge 6](#challenge-6-ngc-pytorch-containers-and-gb10--status-as-of-april-2026) |
| `Target modules X not found in base model` | LoRA config has invalid module names | Inspect model with `model.named_modules()` |
| DataLoader workers use excessive memory | Workers fork entire process state | Set `num_workers=0` |
| `platform (linux/amd64) does not match` | Docker image is x86_64 only | Use ARM64-native image |

---

## Quick Reference: Build Times on DGX Spark

| Package | Build Time | Notes |
|---------|------------|-------|
| mmcv (with CUDA ops) | ~25 minutes | Compiling CUDA kernels |
| Flash Attention 2 | ~15-20 minutes | Building for sm_120 |
| Full Docker image (MMLab stack) | ~30-35 minutes | Including all compilations |

---

## Template: Minimal venv Setup

For simple projects that don't require MMLab:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
pip install soundfile  # For audio loading/saving (TorchCodec workaround)
```

---

## Template: Docker Setup for MMLab Projects

```dockerfile
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="12.0"
ENV MMCV_WITH_OPS=1

RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3-pip \
    git wget curl ffmpeg \
    libsm6 libxext6 libgl1-mesa-glx libglib2.0-0 \
    build-essential

RUN ln -sf /usr/bin/python3.10 /usr/bin/python

RUN pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Build mmcv from source
RUN pip install git+https://github.com/open-mmlab/mmcv.git@v2.1.0 --no-build-isolation
```

---

## Template: docker-compose.yml with GPU Access

```yaml
version: '3.8'

services:
  ml-app:
    build: .
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu, compute, utility]
    volumes:
      - ./models:/app/models
    ports:
      - "7860:7860"

  tensorboard:
    image: python:3.10-slim  # ARM64-native, not tensorflow/tensorflow
    command: >
      bash -c "pip install -q tensorboard && tensorboard --logdir=/logs --host=0.0.0.0 --port=6006"
    volumes:
      - ./logs:/logs
    ports:
      - "6006:6006"
```

---

## Verified Working Configurations

These configurations have been tested and verified working on DGX Spark by Martim Ramos:

### Video Generation (Diffusion Models)
- **Approach:** Python venv with PyTorch nightly
- **VRAM usage:** ~49GB for large models
- **Performance:** Working at full speed

### Lip Sync / Avatar Animation (MMLab-based)
- **Approach:** Docker with Ubuntu 22.04 + Python 3.10
- **Key requirement:** mmcv built from source
- **Status:** Fully functional

### Image/Video Generation (5B+ parameter models)
- **Approach:** Python venv with Flash Attention 2 from source
- **Performance:** ~65 seconds/step (compute-bound)
- **Note:** Some video input features disabled due to decord

### Music Generation (LoRA Training)
- **Approach:** Docker with custom Dockerfile
- **Model size:** 4.5B parameters total, 283M trainable (LoRA)
- **Key fixes applied:**
  - spacy version flexibility (`>=3.7.0`)
  - transformers/peft upgraded (`>=4.51.0` / `>=0.17.0`)
  - Audio I/O patched to use `soundfile`
  - LoRA target modules corrected (`to_q`, `to_k`, `to_v`, `add_q_proj`, etc.)
  - DataLoader set to `num_workers=0`
- **Performance:** ~3.2 seconds/iteration, 96% GPU utilization
- **Status:** Training and inference working

### ComfyUI (image / video diffusion workflows)
- **Approach:** Docker on `nvcr.io/nvidia/pytorch:25.10-py3` (NGC)
- **Image size:** ~30 GB
- **Why NGC:** Ships torch `2.9.0a0+...nv25.10` with sm_120 + compute_120 PTX, CUDA 13.0, NVRTC 13.0 — verified to run `complex.abs()`, `torch.fft.rfft().abs()`, and other jiterator-codegen ops on GB10 with no patching
- **Key fixes applied:**
  - `sed -i '/^torch[>=< ]/d; /^torch$/d; /^torchvision/d; /^torchaudio/d' requirements.txt` before `pip install -r requirements.txt` — preserves NGC's torch while keeping torchsde
  - Non-root user (UID 1100) for safety
  - `deploy.resources.reservations` in compose (not `runtime: nvidia`)
- **Status:** Working in production; full Dockerfile in [Challenge 16](#challenge-16-comfyui-on-dgx-spark-the-official-playbook-is-fragile-use-ngc)

### TTS — Chatterbox Multilingual (zero-shot voice cloning, PT-PT)
- **Approach:** Python venv (3.12) with PyTorch nightly cu128
- **Key fixes applied:**
  - Chatterbox installed with `--no-deps` to keep cu128 nightly (Challenge 15)
  - Reinstalled deps individually; `onnx==1.16.0` pinned for aarch64 wheel
  - System `libnvrtc.so.13` symlinked over bundled `libnvrtc.so.12` (Challenge 14) — fixes runtime crash in `Kaldi.fbank` / `log_mel_spectrogram` during reference encoding
  - `soundfile.write` instead of `torchaudio.save` for output (Challenge 12)
- **Performance:** ~50 sampling steps/sec for the t3 sampler on GB10
- **Verified:** Full PT-PT zero-shot generation from a 15s reference clip; output at 24 kHz
- **Status:** Working end-to-end

---

## Decision Tree: Which Setup Do I Need?

```
Starting a new ML project on DGX Spark?
│
├─ Does it use MMLab (mmcv, mmdet, mmpose)?
│   ├─ YES → Use Docker with Ubuntu 22.04 + Python 3.10
│   └─ NO  → Continue below
│
├─ Does it need Flash Attention?
│   ├─ YES → Build Flash Attention 2 from source (sm_120)
│   └─ NO  → PyTorch SDPA fallback works fine
│
├─ Does it need video decoding (decord)?
│   ├─ YES → Feature not available on ARM64 — find alternatives
│   └─ NO  → Continue below
│
├─ Does it need audio loading/saving?
│   ├─ YES → Use soundfile instead of torchaudio
│   └─ NO  → Continue below
│
├─ Does it run FFT / mel-spectrogram / complex tensors on GPU?
│  (TTS, ASR, audio diffusion all do)
│   ├─ YES → Symlink system NVRTC 13 over bundled NVRTC 12 (Challenge 14)
│   └─ NO  → Continue below
│
├─ Does its pyproject.toml pin torch==2.6.0 (or any older torch)?
│   ├─ YES → pip install --no-deps, then reinstall deps manually (Challenge 15)
│   └─ NO  → Continue below
│
├─ Does it pin specific package versions?
│   ├─ YES → Check ARM64 wheel availability:
│   │        • spacy: use >=3.7.0 if pinned version fails
│   │        • peft: match version to transformers
│   └─ NO  → Continue below
│
├─ Using Docker?
│   ├─ YES → Use deploy.resources.reservations syntax
│   │        Use python:3.10-slim for TensorBoard (not tensorflow/tensorflow)
│   └─ NO  → Continue below
│
└─ Basic setup:
    1. python3 -m venv .venv
    2. pip install --pre torch --index-url .../nightly/cu128
    3. pip install soundfile (if using audio)
    4. Install project dependencies
    5. Patch torch.load if using old checkpoints
    6. Set num_workers=0 if DataLoader causes memory issues
```

---

## Frequently Asked Questions

### Is the DGX Spark good for ML development?

Yes, once you get past the initial setup pain. The 128GB unified memory lets you run models that would require multi-GPU setups elsewhere. After configuration, it's a powerful development machine for local AI work.

### Why isn't NVIDIA providing better software support?

The GB10 is new hardware (Blackwell architecture). NVIDIA is actively working on support — NGC containers and official PyTorch releases will likely add sm_121 support in future updates. This guide bridges the gap until then.

### Should I use Docker or native Python venv?

- **Use Docker** for MMLab projects, complex dependency chains, or reproducible builds
- **Use venv** for simple projects, quick experiments, or when you need faster iteration

### Can I use this guide for other Blackwell GPUs?

Partially. The PyTorch nightly + CUDA 12.8 approach should work for any Blackwell GPU. However, specific challenges around ARM64 wheels are unique to DGX Spark's aarch64 platform.

### Why is my DataLoader using so much memory?

DataLoader workers fork the entire process state. On DGX Spark with large models already loaded, this can exhaust memory quickly. Set `num_workers=0` to load data in the main process.

### Why does my TTS / audio model crash with `nvrtc: error: invalid value for --gpu-architecture`?

The bundled NVRTC 12.8 in PyTorch nightly doesn't know GB10's sm_121 arch. As soon as the jiterator JIT-compiles a kernel for an uncommon type combo (e.g. `.abs()` on a complex tensor returned by `torch.fft.rfft`, used inside `Kaldi.fbank` and most mel-spectrogram code), NVRTC errors out. Symlink the system `libnvrtc.so.13` over the bundled `libnvrtc.so.12` — see [Challenge 14](#challenge-14-runtime-nvrtc-error-bundled-libnvrtcso12-doesnt-know-sm_121).

### Can I run Chatterbox-TTS on DGX Spark?

Yes. Install with `--no-deps` (it pins `torch==2.6.0` which would clobber the cu128 nightly), reinstall dependencies individually, do the NVRTC 13 symlink, and use `soundfile.write` instead of `torchaudio.save`. See the **Chatterbox Multilingual** entry under [Verified Working Configurations](#verified-working-configurations).

### How do I install a Python package whose pyproject.toml pins old PyTorch?

Use `pip install <project> --no-deps`, then reinstall the rest of its dependencies manually, omitting the torch/torchaudio/numpy pins. See [Challenge 15](#challenge-15-project-pins-torch260-and-clobbers-your-cu128-nightly).

### Does the official NVIDIA ComfyUI playbook for DGX Spark actually work?

It works on a clean machine with default ComfyUI and no custom nodes. As soon as you install custom nodes (which is most of why people use ComfyUI), their `requirements.txt` files can silently clobber the torch you installed — leaving you on CPU. Use the NGC + sed approach in [Challenge 16](#challenge-16-comfyui-on-dgx-spark-the-official-playbook-is-fragile-use-ngc) instead.

### Should I use NGC PyTorch or pip-install nightly cu128 on DGX Spark?

NGC `pytorch:25.10-py3`+ is the cleanest path: matches the host's CUDA 13.0 and NVRTC 13.0, ships sm_120 + compute_120 PTX (which JITs to sm_121), and avoids the `--no-deps` and NVRTC-symlink dances entirely. Trade-off: ~30 GB image. Use cu128 nightly + symlink when you need a small image or bleeding-edge features. See [Challenge 6](#challenge-6-ngc-pytorch-containers-and-gb10--status-as-of-april-2026).

### My ComfyUI samplers (DPM++ SDE, DPM2 a) suddenly stop working after a Docker rebuild — why?

You probably stripped `torchsde` along with `torch` from `requirements.txt`. Use the precise regex from [Challenge 16](#challenge-16-comfyui-on-dgx-spark-the-official-playbook-is-fragile-use-ngc): `^torch[>=< ]`, `^torch$`, `^torchvision`, `^torchaudio` — not `grep -v torch`.

### How do I get help if I'm stuck?

1. Check the [PyTorch DGX Spark Discussion Thread](https://discuss.pytorch.org/t/dgx-spark-gb10-cuda-13-0-python-3-12-sm-121/223744)
2. Search NVIDIA Developer Forums for GB10-specific issues
3. File issues on project GitHub repos mentioning "DGX Spark" and "sm_121"

---

## External Resources

- [PyTorch DGX Spark Discussion](https://discuss.pytorch.org/t/dgx-spark-gb10-cuda-13-0-python-3-12-sm-121/223744) — Community thread with ongoing solutions
- [NVIDIA PyTorch Release Notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-02.html) — Official compatibility information
- [NVIDIA Spark CUDA-X Instructions](https://build.nvidia.com/spark/cuda-x-data-science/instructions) — NVIDIA's setup guide

---

## About the Author

- GitHub: [github.com/martimramos](https://github.com/martimramos)
- LinkedIn: [linkedin.com/in/martimramos](https://linkedin.com/in/martimramos)


This guide was written after spending 72+ hours debugging DGX Spark issues across multiple ML projects. If it saved you time, consider starring the repo.

---

## Changelog

| Date | Changes |
|------|---------|
| April 2026 | Added Challenge 16 (ComfyUI on DGX Spark — NGC + precise sed regex; official playbook is fragile) |
| April 2026 | Rewrote Challenge 6 — NGC `pytorch:25.10-py3`+ now supports GB10 (verified: torch `2.9.0a0+...nv25.10`, CUDA 13.0, NVRTC 13.0, compute_120 PTX) |
| April 2026 | Added Challenge 14 (NVRTC bundled lib doesn't know sm_121 — runtime fix for TTS/audio/FFT pipelines) |
| April 2026 | Added Challenge 15 (`--no-deps` recipe for projects pinning `torch==2.6.0`, e.g. Chatterbox-TTS) |
| April 2026 | Added Verified Configurations: Chatterbox Multilingual TTS (PT-PT) and ComfyUI (NGC) |
| April 2026 | Expanded error table, decision tree, and FAQ with NVRTC/jiterator, torch-pin, and ComfyUI scenarios |
| January 2025 | Added Challenge 12 (TorchCodec) and Challenge 13 (TensorBoard ARM64) |
| January 2025 | Added music LoRA training configuration with performance benchmarks |
| January 2025 | Expanded error table with DataLoader memory and LoRA module errors |
| January 2025 | Initial release with 11 documented challenges |

---

## Contributing

Found a new issue? Discovered a better solution? Contributions welcome:

1. Open an issue describing the problem and your DGX Spark configuration
2. Submit a PR with your fix and reproduction steps
3. Help others in the discussions — we're all figuring this out together

---

*This guide is maintained by Martim Ramos. Last verified on NVIDIA DGX Spark with GB10 (Blackwell), April 2026.*
