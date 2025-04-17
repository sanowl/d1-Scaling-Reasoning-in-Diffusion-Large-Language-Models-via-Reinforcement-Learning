from __future__ import annotations
import math, argparse, json, os, time, re, itertools
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model
from trl import AutoModelForSeq2SeqLMWithValueHead
import secrets

DTYPE = torch.bfloat16
ACCEL = Accelerator()


def contains_any(x: torch.Tensor, values: set[int]) -> torch.Tensor:
    """vectorised *isin* that works on PyTorch 1.13+"""
    mask = torch.zeros_like(x, dtype=torch.bool)
    for v in values:
        mask |= (x == v)
    return mask

@torch.no_grad()
def diffuse_decode(model, tok: AutoTokenizer, prompt_ids: torch.Tensor, num_tokens: int, top_p: float = 0.95) -> torch.Tensor:
    B, P = prompt_ids.shape; device = prompt_ids.device
    seq = torch.cat([prompt_ids, torch.full((B, num_tokens), tok.mask_token_id, device=device)], dim=1)
    block_size, steps = 32, num_tokens // 2
    for step in range(steps):
        blk = min((step*2)//block_size, (num_tokens//block_size)-1)
        start,end = P + blk*block_size, P + (blk+1)*block_size
        logits = model(seq).logits[:, start:end, :]
        probs = logits.softmax(-1)
        conf, pred = probs.topk(1, -1)
        conf = conf.squeeze(-1)
        masked = (seq[:, start:end] == tok.mask_token_id)
        conf_masked = conf.clone(); conf_masked[~masked] = -1.0
        idx = conf_masked.view(B,-1).topk(2, -1).indices
        for b in range(B):
            for j in idx[b]:
                seq[b, start + j] = pred[b, j, 0]
        if (seq[:, P:] != tok.mask_token_id).all():
            break
    return seq[:, P:]

@torch.no_grad()
def one_step_logp(model, tok: AutoTokenizer, prompt: torch.Tensor, comp: torch.Tensor, p: float) -> Tuple[torch.Tensor, torch.Tensor]:
    special = {tok.bos_token_id, tok.eos_token_id, tok.pad_token_id}
    mask = (torch.rand_like(prompt.float()) < p) & (~contains_any(prompt, special))
    q_prime = prompt.clone(); q_prime[mask] = tok.mask_token_id
    inp = torch.cat([q_prime, comp.full_like(comp, tok.mask_token_id)], 1)
    logits = model(inp).logits[:, -comp.size(1):, :]
    tok_logp = F.log_softmax(logits, -1).gather(2, comp.unsqueeze(-1)).squeeze(-1)
    seq_logp = tok_logp.sum(-1)
    return tok_logp, seq_logp

class MaskedSFT:
    def __init__(self, model, tok: AutoTokenizer, args):
        self.tok, self.args = tok, args
        self.model = model
        if args.use_lora:
            self.model = get_peft_model(self.model, LoraConfig(r=8,lora_alpha=16,lora_dropout=0.05,
                                                               target_modules=["q_proj","k_proj","v_proj","o_proj"],
                                                               task_type="CAUSAL_LM"))
        self.model.resize_token_embeddings(len(tok))
        self.model.gradient_checkpointing_enable()
        self.model.to(ACCEL.device, dtype=DTYPE)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=args.lr_sft, betas=(0.9,0.99), weight_decay=0.1)

    def _mask(self, ex):
        p_ids = self.tok(ex["prompt"], truncation=True,max_length=self.args.max_len, add_special_tokens=False)["input_ids"]
        r_ids = self.tok(ex["response"],truncation=True,max_length=self.args.max_len, add_special_tokens=False)["input_ids"]
        t = secrets.SystemRandom().random(); alpha = 1.-t
        masked = [tid if secrets.SystemRandom().random()<alpha else self.tok.mask_token_id for tid in r_ids]
        return {"input_ids": p_ids+masked, "labels": r_ids}

    def fit(self, ds):
        ds = ds.map(self._mask, remove_columns=ds.column_names)
        dl = DataLoader(ds, batch_size=self.args.bs_sft, shuffle=True,
                        collate_fn=lambda b: self.tok.pad(b, return_tensors="pt", padding_value=self.tok.pad_token_id))
        total = self.args.epochs_sft*len(dl)
        sched = get_cosine_schedule_with_warmup(self.opt, 50, total)
        grad_acc = self.args.grad_acc_sft
        self.model.train()
        for epoch in range(self.args.epochs_sft):
            for i,batch in enumerate(dl):
                batch = {k:v.to(ACCEL.device) for k,v in batch.items()}
                loss = self.model(**batch).loss / grad_acc
                ACCEL.backward(loss)
                if (i+1)%grad_acc==0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),0.1)
                    self.opt.step(); sched.step(); self.opt.zero_grad()
            ACCEL.print(f"[SFT] epoch {epoch+1} done")
        self.model.save_pretrained(self.args.out_dir/"sft")


class DiffuGRPO:
    def __init__(self, model_path: str|Path, tok: AutoTokenizer, args):
        self.tok, self.args = tok, args
        self.model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(model_path).to(ACCEL.device, dtype=DTYPE)
        if args.use_lora:
            self.model.base_model = get_peft_model(self.model.base_model, LoraConfig(r=128,lora_alpha=64,lora_dropout=0.05,
                                                                                    target_modules=["q_proj","k_proj","v_proj","o_proj"],
                                                                                    task_type="CAUSAL_LM"))
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=args.lr_rl, betas=(0.9,0.99), weight_decay=0.1)

    @staticmethod
    def _boxed(sol:str):
        m=re.search(r"\\\\boxed\\{([^}]*)\\}", sol); return m.group(1).strip() if m else ""

    def rew_gsm(self, comp:str,target:str):
        try: ans=int(comp.split("<answer>")[1].split("</answer>")[0])
        except: ans=None
        xml=0.125*(comp.count("<reasoning>")+comp.count("</reasoning>")+comp.count("<answer>")+comp.count("</answer>"))
        soft=0.5 if "<reasoning>" in comp and "<answer>" in comp else 0
        strict=0.5 if comp.strip().startswith("<reasoning>") and "</reasoning><answer>" in comp else 0
        intok=0.5 if ans is not None else 0
        corr=2.0 if (ans is not None and ans==int(target)) else 0
        return xml+soft+strict+intok+corr

    def rew_math(self, comp:str,boxed:str):
        has_box = f"\\boxed{{{boxed}}}" in comp
        fmt = 1.0 if has_box else 0.75 if "<answer>" in comp else 0.25
        return fmt + (2.0 if has_box else 0.0)

    def rew_count(self, comp:str, tgt:int, nums:List[int]):
        try: val = int(eval(comp.split("=")[0]))
        except: return 0.0
        uses = all(comp.count(str(n))==1 for n in nums)
        if uses and val==tgt: return 1.0
        if uses: return 0.1
        return 0.0

    def rew_sudoku(self, comp:str, sol:str):
        pred = re.sub(r"\D","",comp)[:16]
        if len(pred)!=16: return 0.0
        correct=sum(1 for p,s in zip(pred,sol) if s!='0' and p==s)
        total = sum(1 for s in sol if s!='0')
        return correct/total if total else 0.0

    # task‑agnostic wrapper
    def reward(self, task:str, comp:str, meta:Dict[str,Any]):
        if task=="gsm8k": return self.rew_gsm(comp, meta["target"])
        if task=="math500":return self.rew_math(comp, meta["target"])
        if task=="count":  return self.rew_count(comp, meta["target"], meta["nums"])
        if task=="sudoku": return self.rew_sudoku(comp, meta["target"])
        return 0.0

    def outer_step(self, prompts: torch.Tensor, metas: List[Dict[str,Any]], task:str):
        B=prompts.size(0)
        comps = diffuse_decode(self.model.base_model, self.tok, prompts, self.args.gen_len)
        comp_txt = self.tok.batch_decode(comps, skip_special_tokens=True)
        rewards = torch.tensor([self.reward(task, c,m) for c,m in zip(comp_txt, metas)], device=ACCEL.device)
        old_lp,_=one_step_logp(self.model.base_model, self.tok, prompts, comps, self.args.pmask)
        adv=(rewards-rewards.mean()).unsqueeze(1).expand_as(old_lp)
        self.model.train()
        for _ in range(self.args.mu):
            new_lp,_=one_step_logp(self.model.base_model, self.tok, prompts, comps, self.args.pmask)
            ratio=torch.exp(new_lp-old_lp.detach())
            loss=-torch.minimum(ratio*adv, torch.clamp(ratio,1-0.2,1+0.2)*adv).mean()
            kl=F.kl_div(new_lp,old_lp.detach(),log_target=True,reduction="batchmean")
            total=loss+self.args.beta*kl
            ACCEL.backward(total/self.args.mu)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),0.2)
        self.opt.step(); self.opt.zero_grad()

    def fit(self, ds, task:str):
        def coll(batch):
            pr=[x["prompt_ids"] for x in batch]
            padded=torch.nn.utils.rnn.pad_sequence(pr,batch_first=True,padding_value=self.tok.pad_token_id)
            metas=[x["meta"] for x in batch]
            return {"prompt_ids":padded, "meta":metas}
        dl=DataLoader(ds, batch_size=self.args.bs_rl, shuffle=True, collate_fn=coll)
        steps=0
        for epoch in itertools.count():
            for batch in dl:
                self.outer_step(batch["prompt_ids"].to(ACCEL.device), batch["meta"], task)
                steps+=1
                if steps%100==0: ACCEL.print(f"[RL] step {steps}/{self.args.steps_rl}")
                if steps>=self.args.steps_rl:
                    self.model.save_pretrained(self.args.out_dir/f"d1-{task}")
                    return


def s1k(tok):
    ds=load_dataset("Muennighoff/s1k", split="train")
    return ds

def gsm(tok,split="train"):
    ds=load_dataset("openai/gsm8k", split=split)
    def _p(e):
        return {"prompt":e["question"],"response":e["answer"],"prompt_ids":tok(e["question"],return_tensors="pt").input_ids[0],
                "meta":{"target":e["answer"].split("####")[-1].strip()}}
    return ds.map(_p, remove_columns=ds.column_names)

def math500(tok,split="train"):
    ds=load_dataset("ankner/math-500", split=split)
    def _p(e):
        boxed=DiffuGRPO._boxed(e["solution"]) or e["solution"].split()[-1]
        return {"prompt":e["problem"],"response":e["solution"],"prompt_ids":tok(e["problem"],return_tensors="pt").input_ids[0],
                "meta":{"target":boxed}}
    return ds.map(_p, remove_columns=ds.column_names)

def countdown3(tok,split="train"):
    ds=load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split=split).filter(lambda x: len(x["numbers"])==3)
    def _p(e):
        nums=e["numbers"]
        tgt=e["target"]
        prompt=f"Numbers: {nums}\nTarget: {e['target']}\nGive one expression = target."
        return {"prompt_ids":tok(prompt,return_tensors="pt").input_ids[0], "meta":{"target":tgt,"nums":nums}}
    return ds.map(_p, remove_columns=ds.column_names)

def sudoku4(tok,limit=10000):
    ds=load_dataset("Black-Phoenix/4x4-Sudoku-Dataset", split="train").shuffle(seed=42).select(range(limit))
    def _p(e):
        grid=e["puzzle"]; sol=e["solution"]
        rows="\n".join([grid[i:i+4] for i in range(0,16,4)])
        prompt=f"Solve 4x4 Sudoku:\n{rows}"
        return {"prompt_ids":tok(prompt,return_tensors="pt").input_ids[0],"meta":{"target":sol}}
    return ds.map(_p, remove_columns=ds.column_names)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="llada/llada-8b-instruct")
    p.add_argument("--out_dir", type=Path, default=Path("./ckpt"))
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--task", choices=["gsm8k", "math500", "count", "sudoku"], default="gsm8k")
    
    # SFT params
    p.add_argument("--epochs_sft", type=int, default=20)
    p.add_argument("--bs_sft", type=int, default=4)
    p.add_argument("--grad_acc_sft", type=int, default=4)
    p.add_argument("--lr_sft", type=float, default=1e-5)
    p.add_argument("--max_len", type=int, default=512)
    
    # RL params
    p.add_argument("--steps_rl", type=int, default=1000)
    p.add_argument("--bs_rl", type=int, default=8)
    p.add_argument("--lr_rl", type=float, default=1e-6)
    p.add_argument("--gen_len", type=int, default=256)
    p.add_argument("--pmask", type=float, default=0.15)
    p.add_argument("--mu", type=int, default=4)
    p.add_argument("--beta", type=float, default=0.01)
    
    args = p.parse_args()
    args.out_dir.mkdir(exist_ok=True)
    
    # Initialize model and tokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    
    # Train
    if args.task == "gsm8k":
        ds = gsm(tok)
    elif args.task == "math500":
        ds = math500(tok)
    elif args.task == "count":
        ds = countdown3(tok)
    elif args.task == "sudoku":
        ds = sudoku4(tok)
        
    sft = MaskedSFT(model, tok, args)
    sft.fit(ds)
    
    rl = DiffuGRPO(args.out_dir/"sft", tok, args)
    rl.fit(ds, args.task)

if __name__ == "__main__":
    main()
