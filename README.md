# qwen-local

One script to get Qwen3 running locally — with a chat UI and optional Claude Code integration.

**What it does:**
- Detects your GPU and picks the best Qwen3 model that fits in your VRAM
- Installs [Ollama](https://ollama.com) if needed
- Downloads your chosen model (with your approval)
- Starts a zero-dependency chat UI in your browser
- Wires up [Claude Code](https://claude.ai/code) to run against your local model — free, offline, no API key

---

## Quickstart

```bash
git clone https://github.com/rhowardstone/qwen-local
cd qwen-local
bash setup.sh
```

That's it. Follow the prompts.

---

## Requirements

| Platform | Requirements |
|----------|-------------|
| **Linux** | curl, GPU drivers (NVIDIA/AMD optional) |
| **macOS** | Homebrew, Apple Silicon or NVIDIA eGPU |
| **Windows** | WSL2 with Ubuntu — run everything inside WSL |

No Python, Node, Docker, or package manager needed beyond the above.

---

## Model guide

The script auto-picks based on your VRAM, but here's the full picture:

| Model | Download | Min VRAM | Speed | Quality |
|-------|----------|----------|-------|---------|
| `qwen3:1.7b` | 1.4 GB | 2 GB | ⚡⚡⚡⚡ | ★★☆☆ |
| `qwen3:4b` | 2.6 GB | 3 GB | ⚡⚡⚡ | ★★★☆ |
| `qwen3:8b` | 5.2 GB | 6 GB | ⚡⚡ | ★★★★ ← sweet spot |
| `qwen3:14b` | 9.3 GB | 10 GB | ⚡ | ★★★★½ |
| `qwen3:32b` | 20.5 GB | 24 GB | 🐢 | ★★★★★ |

**Apple Silicon** uses unified memory — a 32 GB M3 can run `qwen3:14b` comfortably.

If your VRAM is smaller than the model, Ollama will offload layers to RAM — it still works, just slower.

---

## Chat UI

The script starts a local web server and opens `chat.html`. Features:

- Streaming responses with live token counter
- **Thinking mode** — see Qwen3's chain-of-thought (collapsible)
- **Thinking budget** — cap reasoning tokens so it doesn't spiral (default: 1024)
- KaTeX math rendering
- Syntax-highlighted code blocks with copy button
- Multi-turn conversation history
- Model picker (all your Ollama models)

To start it manually later:

```bash
cd qwen-local
python3 -m http.server 8080
# open http://localhost:8080/chat.html
```

---

## Claude Code integration

The setup script creates Ollama aliases that match Claude's model names, then you just set two env vars:

```bash
source .env.qwen-local && claude
```

Or inline:

```bash
ANTHROPIC_BASE_URL=http://localhost:11434 \
ANTHROPIC_AUTH_TOKEN=ollama \
claude
```

**What this does:** Claude Code normally calls `api.anthropic.com`. Setting `ANTHROPIC_BASE_URL` redirects those calls to your local Ollama instance instead. Requires Ollama ≥ 0.14.0 (the script handles this).

**Honest expectations:**

| Task | qwen3:8b | qwen3:14b |
|------|----------|-----------|
| Explaining code | ✅ Great | ✅ Great |
| Writing functions | ✅ Good | ✅ Great |
| Multi-file refactors | ⚠️ Inconsistent | ✅ Good |
| Autonomous agentic tasks | ⚠️ Unreliable | ✅ Usable |

For serious agentic work, `qwen3:14b` is the minimum. `qwen3:8b` is excellent for chat and single-function tasks.

---

## Manual Ollama aliases (if you skipped setup)

```bash
ollama cp qwen3:8b  claude-sonnet-4-6
ollama cp qwen3:14b claude-opus-4-5
ollama cp qwen3:8b  claude-haiku-4-5
```

---

## Uninstall

```bash
# Remove the Ollama aliases
ollama rm claude-sonnet-4-6
ollama rm claude-opus-4-5
ollama rm claude-haiku-4-5

# Remove the models (frees disk space)
ollama rm qwen3:8b
ollama rm qwen3:14b
```

Ollama itself: see [ollama.com/docs](https://ollama.com).

---

## Troubleshooting

**"Cannot connect to Ollama"** — run `ollama serve` in a separate terminal.

**Very slow generation** — your model is larger than your VRAM; it's offloading to RAM. Try a smaller model.

**Claude Code still uses Anthropic** — make sure `ANTHROPIC_BASE_URL` is exported in the same shell you launch `claude` from. Check with `echo $ANTHROPIC_BASE_URL`.

**WSL2 GPU not detected** — ensure you have [WSL2 with GPU support](https://learn.microsoft.com/en-us/windows/wsl/tutorials/gpu-compute) and NVIDIA drivers installed on the Windows side.
