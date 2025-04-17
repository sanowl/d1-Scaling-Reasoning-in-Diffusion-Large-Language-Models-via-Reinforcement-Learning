# d1‑reference   <sup>(Masked‑SFT ✚ diffu‑GRPO)</sup>

This repository is a **faithful, runnable replication** of the paper:<br>
**“d1: Scaling Reasoning in Diffusion Large Language Models via Reinforcement Learning”** (Zhao *et al.*, ICLR 2025).

The code lives in [`d1_full.py`](./d1_full.py) and implements every component end‑to‑end:

| Stage | Paper section | File object | Notes |
|-------|---------------|-------------|-------|
| 1. Masked SFT | Algorithm 2, Sec 3.3 | `MaskedSFT` class | linear noise schedule $\alpha_t = 1-t$, LoRA $r{=}8$ |
| 2. diffu‑GRPO | Algorithm 1, Eq (2‑4) | `DiffuGRPO` class | one‑step log‑$p$ estimator, random prompt masking $p_{mask}=0.15$ |
| Decoder | Appendix B (Inference) | `diffuse_decode()` | semi‑AR 2‑token unmasking, block = 32 |

---
## 1 Mathematical foundations

### 1.1 Masked diffusion objective
For a sequence $x_0$ and linear schedule $\alpha_t=1-t$, the NELBO to minimise during pre‑training or SFT is

$$
\mathcal L_{\text{NELBO}}(\theta)\;=\;\mathbb E_{t\sim\mathcal U[0,1)}\,\mathbb E_{x_0,\,x_t\sim q_{t|0}}\Bigg[\;\frac1t \sum_{k=1}^{|x_t|}\mathbf 1[x^k_t = \texttt{<mask>}]\,\log f_\theta\bigl(x^k_0 \mid x_t\bigr)\Bigg].
$$

### 1.2 One‑step log‑probability estimator (Sec 3.1)
Given *partially* masked prompt $q'$, the per‑token log‑prob is approximated by a **single forward pass**:

$$
\log \pi_\theta(o_k\mid q)\;\approx\;\log f_\theta\bigl(o_k\mid q'\,\Vert\,\underbrace{\texttt{<mask>}\,\dots\,\texttt{<mask>}}_{|o|}\bigr).
$$

The sequence log‑prob uses a mean‑field factorisation $\log\pi(o|q)\approx\sum_k\log\pi(o_k|q)$.

### 1.3 diffu‑GRPO loss

For completions $o_i$ (group size $G$) and advantages $A_i^k = r_i - \frac{1}{G}\sum_j r_j$ we minimise:

$$
\mathcal{L} = \mathbb{E}_{q,o}\left[\frac{1}{G}\sum_{i=1}^G\frac{1}{|o_i|}\sum_{k} \min\left(\rho_i^k A_i^k,\,\operatorname{clip}(\rho_i^k,1-\epsilon,1+\epsilon)A_i^k\right)\right] - \beta D_{\mathrm{KL}}\left(\pi_\theta\,\|\,\pi_{\text{ref}}\right)
$$

with importance ratio $\rho_i^k = \exp(\log\pi_\theta - \log\pi_{\text{old}})$

---
## 2 Environment setup
```bash
conda create -n d1 python=3.10
conda activate d1
pip install torch==2.2.0+cu118 torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate datasets trl peft bitsandbytes flash-attn==2.5.2 rich
```
The script will download **LLaDA‑8B‑Instruct** automatically from Hugging Face.

---
## 3 Running experiments
### 3.1 Stage 1 – Masked SFT
```bash
python d1_full.py \
  --model llada/llada-8b-instruct \
  --use_lora             # LoRA r=8 α=16
  --task sft_only        # skips RL and saves to ./ckpt/sft
```
### 3.2 Stage 2 – RL (diffu‑GRPO)
Train **per task** (GSM8K shown):
```bash
python d1_full.py \
  --model ./ckpt/sft \
  --use_lora \
  --task gsm8k \
  --steps_rl 7700 --bs_rl 6 --mu 12 --pmask 0.15
```
Set `--task` to `math500`, `count`, or `sudoku` for the other benchmarks. Hyper‑parameters default to the paper.

---
## 4 Reproducing Table 1
| Task | RL steps | Expected acc (512 tokens) |
|------|---------:|---------------------------:|
| GSM8K | 7 700 | **82.1 %** |
| MATH500 | 6 600 | 40.2 % |
| Countdown‑3 | 5 000 | 42.2 % |
| Sudoku‑4×4 | 3 800 | 22.1 % |

GPU budget matches the paper (8× A100‑80GB, Flash‑Attention 2, 4‑bit quant).

---
## 5 Citation
```
@inproceedings{zhao2025d1,
  title     = {d1: Scaling Reasoning in Diffusion Large Language Models via Reinforcement Learning},
  author    = {Siyan Zhao and Devaansh Gupta and Qinqing Zheng and Aditya Grover},
  booktitle = {International Conference on Learning Representations},
  year      = {2025}
}
```

