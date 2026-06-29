#!/usr/bin/env python3
"""
Probe Server — Qwen3 inference with per-token hidden state probing.

Replaces Ollama. Streams tokens + logprobs + probe scores per token.
Compatible with the qwen-local chat UI (same /api/chat endpoint shape).

Install:
  pip install fastapi uvicorn transformers bitsandbytes accelerate torch

Run:
  python server.py
  python server.py --model Qwen/Qwen3-8B --port 8000 --probe-layer 18
"""

import argparse
import asyncio
import json
import sys

import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ── CLI ──────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--model",       default="Qwen/Qwen3-8B")
p.add_argument("--port",        type=int,   default=8000)
p.add_argument("--probe-layer", type=int,   default=-1,
               help="Transformer layer index to probe (-1 = auto: middle layer)")
p.add_argument("--no-quant",    action="store_true",
               help="Disable 4-bit quantization (needs 16 GB+ VRAM)")
p.add_argument("--temperature", type=float, default=0.6)
p.add_argument("--top-p",       type=float, default=0.9)
args = p.parse_args()

# ── Globals ──────────────────────────────────────────────────────────────────
model     = None
tokenizer = None
probe_layer_idx = args.probe_layer   # resolved after model load
probes: dict[str, torch.Tensor] = {}  # name → unit vector (CPU fp32)

# ── Probe contrast-pair definitions ─────────────────────────────────────────

POSITIVE_TEMPLATES = {
    "truthfulness": [
        "The following statement is true: {s}",
        "This is a verified fact: {s}",
        "It is factually correct that {s}",
        "This is accurate information: {s}",
    ],
    "positivity": [
        "This is a positive and uplifting thought: {s}",
        "Something good and wonderful happened: {s}",
        "A happy, optimistic outlook: {s}",
    ],
    "certainty": [
        "I am completely certain that {s}",
        "There is no doubt that {s}",
        "This is definitively established: {s}",
    ],
}

NEGATIVE_TEMPLATES = {
    "truthfulness": [
        "The following statement is false: {s}",
        "This is misinformation: {s}",
        "It is factually incorrect that {s}",
        "This is inaccurate information: {s}",
    ],
    "positivity": [
        "This is a negative and pessimistic thought: {s}",
        "Something bad and disappointing happened: {s}",
        "A sad, gloomy outlook: {s}",
    ],
    "certainty": [
        "I am very uncertain whether {s}",
        "It is highly debatable whether {s}",
        "This is entirely speculative: {s}",
    ],
}

PROBE_STATEMENTS = {
    "truthfulness": [
        "the sky appears blue due to Rayleigh scattering",
        "water is composed of hydrogen and oxygen",
        "Paris is the capital of France",
        "humans have 46 chromosomes",
        "the Earth orbits the Sun",
        "the speed of light is approximately 300,000 km/s",
        "the Great Wall of China is visible from space with the naked eye",
        "humans only use 10 percent of their brains",
        "antibiotics work against viral infections",
        "lightning never strikes the same place twice",
    ],
    "positivity": [
        "today was a great day",
        "I accomplished my goals",
        "the future looks bright",
        "people are fundamentally kind",
        "it was a terrible outcome",
        "nothing worked as planned",
        "the situation is hopeless",
        "everything fell apart",
    ],
    "certainty": [
        "two plus two equals four",
        "the Earth is spherical",
        "free will exists",
        "consciousness arises from brain activity",
        "this stock will rise tomorrow",
        "life exists elsewhere in the universe",
    ],
}


# ── Probe computation ─────────────────────────────────────────────────────────

@torch.no_grad()
def mean_hidden(texts: list[str]) -> torch.Tensor:
    """Mean-pool hidden states at probe_layer_idx for a batch of texts."""
    enc = tokenizer(texts, return_tensors="pt", padding=True,
                    truncation=True, max_length=256)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    out = model(**enc, output_hidden_states=True)
    h = out.hidden_states[probe_layer_idx]   # [B, L, H]
    mask = enc["attention_mask"].unsqueeze(-1).float()
    pooled = (h * mask).sum(1) / mask.sum(1)  # [B, H]
    return pooled.float().cpu()


def make_probe(pos_texts: list[str], neg_texts: list[str]) -> torch.Tensor:
    pos = mean_hidden(pos_texts).mean(0)
    neg = mean_hidden(neg_texts).mean(0)
    return F.normalize(pos - neg, dim=0)


def bootstrap_builtin_probes():
    for name in ("truthfulness", "positivity", "certainty"):
        print(f"  computing '{name}' probe …", flush=True)
        stmts   = PROBE_STATEMENTS[name]
        pos_tmp = POSITIVE_TEMPLATES[name]
        neg_tmp = NEGATIVE_TEMPLATES[name]
        pos = [t.format(s=s) for s in stmts for t in pos_tmp]
        neg = [t.format(s=s) for s in stmts for t in neg_tmp]
        # batch in chunks of 8 to avoid OOM
        chunk = 8
        pos_vecs = [mean_hidden(pos[i:i+chunk]) for i in range(0, len(pos), chunk)]
        neg_vecs = [mean_hidden(neg[i:i+chunk]) for i in range(0, len(neg), chunk)]
        pos_mean = torch.cat(pos_vecs).mean(0)
        neg_mean = torch.cat(neg_vecs).mean(0)
        probes[name] = F.normalize(pos_mean - neg_mean, dim=0)
    print("Probes ready.", flush=True)


def make_concept_probe(concept: str) -> torch.Tensor:
    pos = [
        f"This text is about {concept}.",
        f"The concept of {concept} is central here.",
        f"Discussing {concept} and related ideas.",
        f"The topic is {concept}.",
        f"{concept} is the main subject.",
    ]
    neg = [
        f"This text has nothing to do with {concept}.",
        f"No mention of {concept} here.",
        f"Completely unrelated to {concept}.",
        f"The topic is entirely different from {concept}.",
        f"This response avoids {concept}.",
    ]
    return make_probe(pos, neg)


# ── Sampling ──────────────────────────────────────────────────────────────────

def sample_next(logits: torch.Tensor,
                temperature: float, top_p: float) -> tuple[int, float]:
    if temperature == 0:
        tid = int(logits.argmax())
        lp  = float(F.log_softmax(logits, dim=-1)[tid])
        return tid, lp

    logits = logits / temperature
    # nucleus sampling
    sorted_l, sorted_i = torch.sort(logits, descending=True)
    cum_p = torch.cumsum(F.softmax(sorted_l, dim=-1), dim=-1)
    remove = (cum_p - F.softmax(sorted_l, dim=-1)) > top_p
    sorted_l[remove] = -float("inf")
    full = torch.full_like(logits, -float("inf"))
    full[sorted_i] = sorted_l
    probs = F.softmax(full, dim=-1)
    tid   = int(torch.multinomial(probs, 1))
    lp    = float(F.log_softmax(logits, dim=-1)[tid])
    return tid, lp


def top_logprobs(logits: torch.Tensor, k: int = 5) -> list[dict]:
    lp, ids = torch.topk(F.log_softmax(logits, dim=-1), k)
    return [{"token": tokenizer.decode([int(i)]), "logprob": float(l)}
            for i, l in zip(ids, lp)]


# ── Generation ────────────────────────────────────────────────────────────────

# EOS token IDs to stop at (filled after model load)
EOS_IDS: set[int] = set()
EOS_STRS = ("</s>", "<|im_end|>", "<|endoftext|>")


async def stream_generate(
    messages:      list[dict],
    max_tokens:    int   = 2048,
    temperature:   float = 0.6,
    top_p:         float = 0.9,
    think:         bool  = True,
    active_probes: list[str] | None = None,
):
    if active_probes is None:
        active_probes = list(probes.keys())

    # Build prompt
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=think,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    pkv       = None
    gen_ids: list[int] = []
    decoded_so_far = ""

    think_open  = False
    think_close = False

    for _ in range(max_tokens):
        with torch.no_grad():
            if pkv is None:
                out = model(input_ids=input_ids,
                            output_hidden_states=True, use_cache=True, return_dict=True)
            else:
                out = model(input_ids=input_ids[:, -1:],
                            past_key_values=pkv,
                            output_hidden_states=True, use_cache=True, return_dict=True)

        pkv    = out.past_key_values
        logits = out.logits[0, -1, :]                        # [vocab]
        hidden = out.hidden_states[probe_layer_idx][0, -1, :].float().cpu()  # [H]

        tid, lp = sample_next(logits, temperature, top_p)
        tlp     = top_logprobs(logits)

        # Probe scores
        scores = {
            name: float(hidden @ probes[name])
            for name in active_probes if name in probes
        }

        # Decode
        gen_ids.append(tid)
        # Decode full sequence for accurate text (handles multi-byte tokens)
        full_text = tokenizer.decode(gen_ids, skip_special_tokens=False,
                                     clean_up_tokenization_spaces=False)
        token_text = full_text[len(decoded_so_far):]
        decoded_so_far = full_text

        # Thinking state
        if not think_open and "<think>" in full_text:
            think_open = True
        if think_open and not think_close and "</think>" in full_text:
            think_close = True

        is_thinking = think_open and not think_close
        # Strip structural markers from display text
        display = token_text
        for marker in ("<think>", "</think>"):
            display = display.replace(marker, "")

        chunk = {
            "message": {
                "role":     "assistant",
                "content":  display if not is_thinking else "",
                "thinking": display if is_thinking else "",
            },
            "logprobs": [{"token": token_text, "logprob": lp, "top_logprobs": tlp}],
            "probe_scores": scores,
            "done": False,
        }
        yield json.dumps(chunk) + "\n"
        await asyncio.sleep(0)   # yield to event loop

        # Stop conditions
        if tid in EOS_IDS:
            break
        if any(s in full_text[-20:] for s in EOS_STRS):
            break

        input_ids = torch.tensor([[tid]], device=model.device)

    yield json.dumps({"done": True, "message": {"role": "assistant", "content": ""}}) + "\n"


# ── FastAPI ───────────────────────────────────────────────────────────────────

app = FastAPI(title="Probe Server")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
async def startup():
    global model, tokenizer, probe_layer_idx, EOS_IDS

    print(f"Loading {args.model} …", flush=True)
    bnb = None if args.no_quant else BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model     = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        quantization_config=bnb,
        torch_dtype=torch.float16 if args.no_quant else None,
        trust_remote_code=True,
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    # hidden_states tuple: [embedding, layer_0, ..., layer_N]  → length N+2
    # probe_layer_idx into that tuple: 0=embed, 1..N+1=transformer layers
    if args.probe_layer == -1:
        probe_layer_idx = (n_layers // 2) + 1  # middle transformer layer
    else:
        probe_layer_idx = min(args.probe_layer + 1, n_layers + 1)

    print(f"Model ready. {n_layers} transformer layers. "
          f"Probing at hidden_states[{probe_layer_idx}] "
          f"(transformer layer {probe_layer_idx - 1}).", flush=True)

    # EOS ids
    for tok in EOS_STRS:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(tid, int) and tid != tokenizer.unk_token_id:
            EOS_IDS.add(tid)
    if tokenizer.eos_token_id:
        EOS_IDS.add(tokenizer.eos_token_id)

    print("Bootstrapping built-in probes …", flush=True)
    # Run in thread pool so startup doesn't block event loop
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, bootstrap_builtin_probes)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/tags")
async def list_models():
    return {"models": [{"name": args.model, "model": args.model}]}


@app.get("/api/probes")
async def list_probes():
    return {"probes": list(probes.keys())}


@app.post("/api/probes/compute")
async def compute_probe(body: dict):
    concept = (body.get("concept") or "").strip()
    if not concept:
        return {"error": "concept required"}
    name = f"concept:{concept}"
    loop = asyncio.get_event_loop()
    probes[name] = await loop.run_in_executor(None, make_concept_probe, concept)
    return {"name": name, "status": "ok"}


@app.delete("/api/probes/{name:path}")
async def delete_probe(name: str):
    if name in probes and name not in ("truthfulness", "positivity", "certainty"):
        del probes[name]
        return {"status": "deleted"}
    return {"error": "not found or built-in"}


@app.post("/api/chat")
async def chat(body: dict):
    messages      = body.get("messages", [])
    opts          = body.get("options", {})
    max_tokens    = opts.get("num_predict", 2048)
    temperature   = opts.get("temperature", args.temperature)
    top_p         = opts.get("top_p", args.top_p)
    think         = body.get("think", True)
    active_probes = body.get("active_probes")   # None = all

    return StreamingResponse(
        stream_generate(messages, max_tokens, temperature, top_p, think, active_probes),
        media_type="application/x-ndjson",
    )


if __name__ == "__main__":
    print(f"Starting probe server on port {args.port}")
    print("  Quantization:", "disabled" if args.no_quant else "4-bit NF4")
    print(f"  Model:   {args.model}")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
