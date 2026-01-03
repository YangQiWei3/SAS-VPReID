# SAS-VPReID: A Scale-Adaptive Framework with Shape Priors for Video-based Person Re-Identification at Extreme Far Distances

Official implementation of **SAS-VPReID**, a unified solution for **extreme far-distance aerial-ground + cloth-changing** video-based person re-identification (VReID-XFD setting).  
The framework integrates three complementary modules:

- **MEVB**: Memory-Enhanced Visual Backbone (CLIP ViT-L + multi-proxy memory supervision + video-consistent augmentation)
- **MGTM**: Multi-Granularity Temporal Modeling (multi-stride slicing + Mamba-based sequence modeling + learnable scale fusion)
- **PRSD**: Prior-Regularized Shape Dynamics (SMPL shape parameters dynamics + explicit shape prior + temporal aggregation)

This method achieves **1st place** on the VReID-XFD challenge leaderboard and strong results on DetReIDXV1 evaluation settings (A→G / G→A / A→A).  
(See paper for details.)  

Paper: *SAS-VPReID: A Scale-Adaptive Framework with Shape Priors for Video-based Person Re-Identification at Extreme Far Distances*.

---

## News
- [ ] Pretrained models will be released.
- [ ] Training & evaluation scripts will be released/cleaned.

---

## Method Overview

### 1) MEVB (Memory-Enhanced Visual Backbone)
- Backbone: **CLIP ViT-L/14** for strong transferability under domain shifts.
- **Video-Consistent Color Jitter**: sample one jitter parameter per tracklet and apply consistently across frames to preserve temporal coherence.
- **Multi-proxy memory**: for each identity maintain multiple proxies updated with momentum; optimize a memory-based contrastive objective.

### 2) MGTM (Multi-Granularity Temporal Modeling)
- Build sequences at multiple temporal strides (e.g., **S = [2, 4, 8]**) to capture both short-term and long-term cues.
- Use an efficient **Mamba-based** temporal operator (bi-directional) and token reordering to emphasize informative content.
- **Learnable fusion** weights adaptively combine features from different strides.

### 3) PRSD (Prior-Regularized Shape Dynamics)
- Regress **SMPL shape parameters (10-dim)** per frame (shape is relatively clothing-invariant).
- Temporal smoothing + Transformer-based aggregation to model shape dynamics under noise.
- Add explicit **SMPL shape prior** regularization for stable training.

Final tracklet descriptor concatenates:
- appearance feature (from backbone pooling),
- temporal feature (MGTM output),
- structural feature (PRSD output).

---

## Environment

Tested with:
- Python >= 3.8
- PyTorch >= 2.0
- CUDA >= 11.7 (recommended)

Main dependencies:
- `torch`, `torchvision`
- `numpy`, `opencv-python`
- CLIP (OpenAI CLIP or equivalent implementation)
- Mamba / selective SSM implementation (if used)
- SMPL utilities (for shape prior and mean-shape parameters)

> Note: exact dependency names may differ depending on the codebase; please check `requirements.txt`.

---

## Installation

```bash
git clone https://github.com/YangQiWei3/SAS-VPReID.git
cd SAS-VPReID

# (recommended) create env
conda create -n sasvpreid python=3.9 -y
conda activate sasvpreid

pip install -r requirements.txt
