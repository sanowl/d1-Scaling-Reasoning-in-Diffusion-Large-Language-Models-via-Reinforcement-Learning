# d1‑reference   <sup>(Masked‑SFT ✚ diffu‑GRPO)</sup>

This repository is a **faithful, runnable replication** of the paper:<br>
**“d1: Scaling Reasoning in Diffusion Large Language Models via Reinforcement Learning”** (Zhao *et al.*, ICLR 2025).

The code lives in [`d1_full.py`](./d1_full.py) and implements every component end‑to‑end. Key stages:

| Stage | Paper section | Implementation | Hyper‑params |
|-------|---------------|----------------|--------------|
| 1. Masked SFT | Alg 2, Sec 3.3 | `MaskedSFT` class | LoRA r=8, α=16, seq 4096, LR 1e‑5, grad clip 0.1 |
| 2. diffu‑GRPO | Alg 1, Eq (2‑4) | `DiffuGRPO` class | pmask=0.15, μ = 12, 4‑bit, LR 5e‑6, β = 0.01 |
| 3. Decoder | App B Inference | `diffuse_decode()` | semi‑AR, 2‑token unmask per step, block=32 |

---
## 1 Mathematical foundations

### 1.1 Masked diffusion objective
Let $x_0$ be a token sequence and adopt the linear schedule $\alpha_t=1-t$. The NELBO is

$$
\mathcal L_{\mathrm{NELBO}}(\theta) = -\mathbb E_{t\sim U[0,1), x_0,x_t\sim q_{t|0}}\Biggl[\frac1t \sum_{k=1}^{|x_t|} \mathbf1[x^k_t=\mathrm{<mask>}]\log f_\theta(x^k_0\mid x_t)\Biggr].
$$

### 1.2 One‑step log‑probability estimator
Perturb the prompt $q\to q'$ by masking each token with prob $p_{mask}$. Then a single forward pass yields:

$$
\log\pi_\theta(o_k\mid q)\approx \log f_\theta\bigl(o_k\mid q',\underbrace{\mathrm{<mask>},\dots,\mathrm{<mask>}}_{|o|}\bigr).
$$

Sequencing uses mean‑field: $\log\pi(o\mid q) \approx \sum_k \log\pi(o_k\mid q)$.

### 1.3 diffu‑GRPO loss
For prompt $q$, sample $G$ completions $o_1,\dots,o_G$. Define per‑token advantage:

$$
A_i^k = r_i - \tfrac1G \sum_{j=1}^G r_j,
$$

and importance ratio:

$$
\rho_i^k = \exp\bigl(\log\pi_\theta(o_{i,k}\mid q) - \log\pi_{\theta_{\mathrm{old}}}(o_{i,k}\mid q)\bigr).
$$

Then:

$$
\mathcal L_{\mathrm{diffu\text{-}GRPO}}(\theta) = \mathbb E_{q, o_{1:G}\sim\pi_{\theta_{\mathrm{old}}}}\Biggl[\frac1G \sum_{i=1}^G\frac1{|o_i|}\sum_{k=1}^{|o_i|} \min\bigl(\rho_i^k A_i^k,\;\mathrm{clip}(\rho_i^k,1-\epsilon,1+\epsilon)A_i^k\bigr)\Biggr] - \beta\,D_{\mathrm{KL}}\bigl(\pi_\theta(\cdot\mid q)\|\pi_{\mathrm{ref}}(\cdot\mid q)\bigr).
$$

---
## 2 Environment setup
```bash
conda create -n d1 python=3.10
conda activate d1
pip install torch==2.2.0+cu118 torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate datasets trl peft bitsandbytes flash-attn==2.5.2 rich
```

---
## 3 Running experiments

### Stage 1 – Masked SFT
```
python d1_full.py \
  --model llada/llada-8b-instruct \
  --use_lora \
  --task sft_only
```

### Stage 2 – diffu‑GRPO (Per benchmark)
Example for GSM8K:
```
python d1_full.py \
  --model ./ckpt/sft \
  --use_lora \
  --task gsm8k \
  --steps_rl 7700 --bs_rl 6 --mu 12 --pmask 0.15
```
Set `--task` to `math500`, `count`, or `sudoku` for the other benchmarks.

---
## 4 Reproducing Table 1
| Task       | RL steps | Gen len |  Expected Acc  |
|------------|---------:|--------:|--------------:|
| GSM8K      | 7 700    | 512     | **82.1 %**    |
| MATH500    | 6 600    | 512     | 40.2 %        |
| Countdown 3| 5 000    | 512     | 42.2 %        |
| Sudoku 4×4 | 3 800    | 512     | 22.1 %        |

---
## 5 Citation
```bibtex
@inproceedings{zhao2025d1,
  title={d1: Scaling Reasoning in Diffusion Large Language Models via Reinforcement Learning},
  author={Siyan Zhao and Devaansh Gupta and Qinqing Zheng and Aditya Grover},
  booktitle={ICLR}, year={2025}
}