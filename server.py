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

Probe vectors are cached to probes_cache.pt after first computation.
Subsequent starts load from cache and skip the ~2 min bootstrapping.
Delete probes_cache.pt to force recomputation.
"""

import argparse
import asyncio
import hashlib
import json
import os
import pathlib

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
p.add_argument("--no-cache",    action="store_true",
               help="Ignore probe cache and recompute from scratch")
args = p.parse_args()

CACHE_FILE = pathlib.Path("probes_cache.pt")

# ── Globals ──────────────────────────────────────────────────────────────────
model           = None
tokenizer       = None
probe_layer_idx = args.probe_layer
probes: dict[str, torch.Tensor] = {}   # name → unit vector (CPU fp32)
probes_status   = "loading"            # "loading" | "computing" | "ready"

# ── Probe contrast-pair definitions ─────────────────────────────────────────

POSITIVE_TEMPLATES = {
    "truthfulness": [
        "The following statement is true: {s}",
        "This is a verified fact: {s}",
        "It is factually correct that {s}",
        "This is accurate: {s}",
    ],
    "positivity": [
        "This is a positive, uplifting thought: {s}",
        "Something good happened: {s}",
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
        "This is inaccurate: {s}",
    ],
    "positivity": [
        "This is a negative, pessimistic thought: {s}",
        "Something bad happened: {s}",
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

# ── Cache key ────────────────────────────────────────────────────────────────

def cache_key() -> str:
    """Fingerprint: model + layer + quant mode. If any changes, recompute."""
    s = f"{args.model}|{probe_layer_idx}|{'noquant' if args.no_quant else '4bit'}"
    return hashlib.md5(s.encode()).hexdigest()[:12]


def load_probe_cache() -> bool:
    if args.no_cache or not CACHE_FILE.exists():
        return False
    try:
        data = torch.load(CACHE_FILE, map_location="cpu", weights_only=True)
        if data.get("key") != cache_key():
            print("Probe cache is stale (model/layer changed) — recomputing.")
            return False
        for name, vec in data["probes"].items():
            probes[name] = vec
        print(f"Loaded {len(probes)} probe vectors from cache ({CACHE_FILE})")
        return True
    except Exception as e:
        print(f"Cache load failed ({e}) — recomputing.")
        return False


def save_probe_cache():
    try:
        # Only cache built-in probes (not ephemeral concept probes)
        builtin = {k: v for k, v in probes.items()
                   if k in ("truthfulness", "positivity", "certainty")}
        torch.save({"key": cache_key(), "probes": builtin}, CACHE_FILE)
        print(f"Probe cache saved to {CACHE_FILE}")
    except Exception as e:
        print(f"Cache save failed: {e}")


# ── Probe computation ─────────────────────────────────────────────────────────

@torch.no_grad()
def mean_hidden(texts: list[str]) -> torch.Tensor:
    enc = tokenizer(texts, return_tensors="pt", padding=True,
                    truncation=True, max_length=256)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    out = model(**enc, output_hidden_states=True)
    h   = out.hidden_states[probe_layer_idx]        # [B, L, H]
    mask = enc["attention_mask"].unsqueeze(-1).float()
    return ((h * mask).sum(1) / mask.sum(1)).float().cpu()  # [B, H]


def make_probe(pos_texts: list[str], neg_texts: list[str]) -> torch.Tensor:
    chunk = 8
    pos = torch.cat([mean_hidden(pos_texts[i:i+chunk]) for i in range(0, len(pos_texts), chunk)])
    neg = torch.cat([mean_hidden(neg_texts[i:i+chunk]) for i in range(0, len(neg_texts), chunk)])
    return F.normalize(pos.mean(0) - neg.mean(0), dim=0)


def bootstrap_builtin_probes():
    global probes_status
    for name in ("truthfulness", "positivity", "certainty"):
        print(f"  [{name}] computing…", flush=True)
        stmts = PROBE_STATEMENTS[name]
        pos = [t.format(s=s) for s in stmts for t in POSITIVE_TEMPLATES[name]]
        neg = [t.format(s=s) for s in stmts for t in NEGATIVE_TEMPLATES[name]]
        probes[name] = make_probe(pos, neg)
        print(f"  [{name}] done", flush=True)
    save_probe_cache()
    probes_status = "ready"
    print("All probes ready — server fully operational.", flush=True)


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
        return tid, float(F.log_softmax(logits, dim=-1)[tid])
    logits = logits / temperature
    sl, si = torch.sort(logits, descending=True)
    cum = torch.cumsum(F.softmax(sl, dim=-1), dim=-1)
    sl[cum - F.softmax(sl, dim=-1) > top_p] = -float("inf")
    full = torch.full_like(logits, -float("inf"))
    full[si] = sl
    probs = F.softmax(full, dim=-1)
    tid   = int(torch.multinomial(probs, 1))
    return tid, float(F.log_softmax(logits, dim=-1)[tid])


def top_logprobs(logits: torch.Tensor, k: int = 5) -> list[dict]:
    lp, ids = torch.topk(F.log_softmax(logits, dim=-1), k)
    return [{"token": tokenizer.decode([int(i)]), "logprob": float(l)}
            for i, l in zip(ids, lp)]


# ── Generation ────────────────────────────────────────────────────────────────

EOS_IDS:  set[int] = set()
EOS_STRS: tuple    = ("</s>", "<|im_end|>", "<|endoftext|>")


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

    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=think)
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    pkv  = None
    gen_ids: list[int] = []
    decoded_so_far = ""
    think_open = think_close = False

    for _ in range(max_tokens):
        with torch.no_grad():
            if pkv is None:
                out = model(input_ids=input_ids,
                            output_hidden_states=True, use_cache=True, return_dict=True)
            else:
                out = model(input_ids=input_ids[:, -1:], past_key_values=pkv,
                            output_hidden_states=True, use_cache=True, return_dict=True)

        pkv    = out.past_key_values
        logits = out.logits[0, -1, :]
        hidden = out.hidden_states[probe_layer_idx][0, -1, :].float().cpu()

        tid, lp = sample_next(logits, temperature, top_p)
        tlp     = top_logprobs(logits)

        # Probe scores — only available probes (may still be computing)
        scores = {name: float(hidden @ probes[name])
                  for name in active_probes if name in probes}

        gen_ids.append(tid)
        full_text  = tokenizer.decode(gen_ids, skip_special_tokens=False,
                                      clean_up_tokenization_spaces=False)
        token_text = full_text[len(decoded_so_far):]
        decoded_so_far = full_text

        if not think_open and "<think>" in full_text:
            think_open = True
        if think_open and not think_close and "</think>" in full_text:
            think_close = True

        is_thinking = think_open and not think_close
        display     = token_text.replace("<think>", "").replace("</think>", "")

        yield json.dumps({
            "message": {
                "role":     "assistant",
                "content":  display if not is_thinking else "",
                "thinking": display if is_thinking     else "",
            },
            "logprobs":    [{"token": token_text, "logprob": lp, "top_logprobs": tlp}],
            "probe_scores": scores,
            "probes_status": probes_status,
            "done": False,
        }) + "\n"

        await asyncio.sleep(0)

        if tid in EOS_IDS:
            break
        if any(s in full_text[-20:] for s in EOS_STRS):
            break

        input_ids = torch.tensor([[tid]], device=model.device)

    yield json.dumps({
        "done": True,
        "message": {"role": "assistant", "content": ""},
        "probes_status": probes_status,
    }) + "\n"


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Probe Server")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
async def startup():
    global model, tokenizer, probe_layer_idx, EOS_IDS, probes_status

    print(f"\nLoading {args.model} …", flush=True)
    bnb = None if args.no_quant else BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto",
        quantization_config=bnb,
        torch_dtype=torch.float16 if args.no_quant else None,
        trust_remote_code=True,
    )
    model.eval()

    n = model.config.num_hidden_layers
    probe_layer_idx = (n // 2) + 1 if args.probe_layer == -1 else min(args.probe_layer + 1, n + 1)
    print(f"Model ready ({n} layers). Probing hidden_states[{probe_layer_idx}] "
          f"(transformer layer {probe_layer_idx-1}).", flush=True)

    for tok in EOS_STRS:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(tid, int) and tid != tokenizer.unk_token_id:
            EOS_IDS.add(tid)
    if tokenizer.eos_token_id:
        EOS_IDS.add(tokenizer.eos_token_id)

    # Try cache first — if hit, server is immediately fully operational
    if load_probe_cache():
        probes_status = "ready"
    else:
        # Bootstrap in background so the server accepts requests immediately
        probes_status = "computing"
        print("Computing probe vectors in background (server is live now)…", flush=True)
        print("Probes will appear in the UI as they finish. "
              "First message can be sent right away.", flush=True)
        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, bootstrap_builtin_probes)

    print(f"\n✓ Server live on http://0.0.0.0:{args.port}  "
          f"(probes: {probes_status})\n", flush=True)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/tags")
async def list_models():
    return {"models": [{"name": args.model, "model": args.model}]}


@app.get("/api/probes")
async def list_probes():
    return {"probes": list(probes.keys()), "status": probes_status}


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
    opts  = body.get("options", {})
    return StreamingResponse(
        stream_generate(
            messages      = body.get("messages", []),
            max_tokens    = opts.get("num_predict", 2048),
            temperature   = opts.get("temperature", args.temperature),
            top_p         = opts.get("top_p", args.top_p),
            think         = body.get("think", True),
            active_probes = body.get("active_probes"),
        ),
        media_type="application/x-ndjson",
    )


if __name__ == "__main__":
    print(f"Probe server  |  model: {args.model}  |  port: {args.port}")
    print(f"Quantization: {'disabled' if args.no_quant else '4-bit NF4'}")
    print(f"Probe cache:  {CACHE_FILE} ({'ignore' if args.no_cache else 'use if present'})\n")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
