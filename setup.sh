#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  qwen-local setup
#  Installs Ollama, picks the best Qwen3 model for your hardware,
#  and wires up Claude Code to run locally for free.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── colours ──────────────────────────────────────────────────────────────────
R='\033[0;31m' G='\033[0;32m' Y='\033[1;33m'
B='\033[0;34m' C='\033[0;36m' W='\033[1m' N='\033[0m'

banner() {
  echo -e "${C}"
  echo "  ╔═══════════════════════════════════════╗"
  echo "  ║           qwen-local setup            ║"
  echo "  ║   local AI on your own GPU, for free  ║"
  echo "  ╚═══════════════════════════════════════╝"
  echo -e "${N}"
}

info()    { echo -e "  ${B}→${N} $*"; }
success() { echo -e "  ${G}✓${N} $*"; }
warn()    { echo -e "  ${Y}!${N} $*"; }
error()   { echo -e "  ${R}✗${N} $*"; exit 1; }
ask()     { echo -e -n "  ${W}?${N} $1 "; }

# ── platform detection ────────────────────────────────────────────────────────
detect_platform() {
  if [[ "$(uname)" == "Darwin" ]]; then
    echo "macos"
  elif grep -qi microsoft /proc/version 2>/dev/null; then
    echo "wsl"
  elif [[ "$(uname)" == "Linux" ]]; then
    echo "linux"
  else
    echo "unknown"
  fi
}

# ── VRAM / memory detection ───────────────────────────────────────────────────
detect_vram_gb() {
  # NVIDIA
  if command -v nvidia-smi &>/dev/null; then
    local mb
    mb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    [[ -n "$mb" && "$mb" =~ ^[0-9]+$ ]] && echo $(( mb / 1024 )) && return
  fi
  # AMD (ROCm)
  if command -v rocm-smi &>/dev/null; then
    local bytes
    bytes=$(rocm-smi --showmeminfo vram 2>/dev/null | awk '/Total Memory/{print $NF}' | head -1)
    [[ -n "$bytes" ]] && echo $(( bytes / 1073741824 )) && return
  fi
  # Intel Arc / generic sysfs
  local sysfs_vram
  sysfs_vram=$(cat /sys/class/drm/card*/device/mem_info_vram_total 2>/dev/null | head -1)
  [[ -n "$sysfs_vram" && "$sysfs_vram" =~ ^[0-9]+$ ]] && echo $(( sysfs_vram / 1073741824 )) && return
  # Apple Silicon — unified memory
  if [[ "$(uname)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
    local bytes
    bytes=$(sysctl -n hw.memsize 2>/dev/null)
    [[ -n "$bytes" ]] && echo $(( bytes / 1073741824 )) && return
  fi
  echo "0"
}

detect_ram_gb() {
  if [[ "$(uname)" == "Darwin" ]]; then
    local bytes
    bytes=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
    echo $(( bytes / 1073741824 ))
  else
    free -g 2>/dev/null | awk '/^Mem:/{print $2}' || echo "0"
  fi
}

detect_gpu_name() {
  if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "NVIDIA GPU"
  elif command -v rocm-smi &>/dev/null; then
    rocm-smi --showproductname 2>/dev/null | awk -F: '/Card series/{print $2}' | head -1 | xargs || echo "AMD GPU"
  elif [[ "$(uname)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
    system_profiler SPHardwareDataType 2>/dev/null | awk '/Chip:/{print $0}' | sed 's/.*Chip: //' || echo "Apple Silicon"
  else
    echo "CPU only"
  fi
}

# ── model selection ───────────────────────────────────────────────────────────
#  Model → Q4 size in GB (approximate)
#    qwen3:1.7b  →  1.4 GB
#    qwen3:4b    →  2.6 GB
#    qwen3:8b    →  5.2 GB
#    qwen3:14b   →  9.3 GB
#    qwen3:32b   → 20.5 GB
# ─────────────────────────────────────────────────────────────────────────────
pick_model() {
  local vram=$1 platform=$2 ram=$3

  if [[ "$platform" == "macos" ]]; then
    # Apple Silicon: unified memory serves as VRAM
    if   [[ $vram -ge 60 ]]; then echo "qwen3:32b"
    elif [[ $vram -ge 20 ]]; then echo "qwen3:14b"
    elif [[ $vram -ge 10 ]]; then echo "qwen3:8b"
    elif [[ $vram -ge  5 ]]; then echo "qwen3:4b"
    else                          echo "qwen3:1.7b"
    fi
  elif [[ $vram -ge 20 ]]; then  echo "qwen3:32b"
  elif [[ $vram -ge  9 ]]; then  echo "qwen3:14b"
  elif [[ $vram -ge  5 ]]; then  echo "qwen3:8b"
  elif [[ $vram -ge  2 ]]; then
    # Partial GPU — use 4b which fits in low-VRAM with CPU offload
    echo "qwen3:4b"
  else
    # CPU-only — pick based on RAM
    if   [[ $ram -ge 16 ]]; then echo "qwen3:8b"
    elif [[ $ram -ge  8 ]]; then echo "qwen3:4b"
    else                         echo "qwen3:1.7b"
    fi
  fi
}

model_size_gb() {
  case "$1" in
    qwen3:1.7b) echo "1.4" ;;
    qwen3:4b)   echo "2.6" ;;
    qwen3:8b)   echo "5.2" ;;
    qwen3:14b)  echo "9.3" ;;
    qwen3:32b)  echo "20.5" ;;
    *)          echo "?" ;;
  esac
}

# ── Ollama install ────────────────────────────────────────────────────────────
install_ollama() {
  local platform=$1
  if command -v ollama &>/dev/null; then
    local ver
    ver=$(ollama --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
    success "Ollama already installed (v${ver})"
    # Require >= 0.14.0 for Anthropic API compat
    local major minor
    major=$(echo "$ver" | cut -d. -f1)
    minor=$(echo "$ver" | cut -d. -f2)
    if [[ $major -eq 0 && $minor -lt 14 ]]; then
      warn "Ollama v${ver} is too old (need ≥ 0.14.0 for Claude Code compat). Upgrading..."
      do_install_ollama "$platform"
    fi
    return
  fi
  info "Installing Ollama..."
  do_install_ollama "$platform"
}

do_install_ollama() {
  local platform=$1
  if [[ "$platform" == "macos" ]]; then
    if command -v brew &>/dev/null; then
      brew install ollama
    else
      error "Homebrew not found. Install it from https://brew.sh then re-run this script.\nOr download Ollama directly from https://ollama.com/download"
    fi
  else
    curl -fsSL https://ollama.com/install.sh | sh
  fi
  success "Ollama installed"
}

# ── ensure Ollama is running ──────────────────────────────────────────────────
ensure_ollama_running() {
  if curl -sf http://localhost:11434/api/tags &>/dev/null; then
    success "Ollama is running"
    return
  fi
  info "Starting Ollama..."
  if command -v systemctl &>/dev/null && systemctl is-active ollama &>/dev/null 2>&1; then
    sudo systemctl start ollama
  else
    ollama serve &>/tmp/ollama-serve.log &
    sleep 3
  fi
  if curl -sf http://localhost:11434/api/tags &>/dev/null; then
    success "Ollama started"
  else
    error "Could not start Ollama. Try running 'ollama serve' in another terminal."
  fi
}

# ── setup Claude Code aliases in Ollama ──────────────────────────────────────
setup_claude_aliases() {
  local model=$1
  info "Creating Claude Code model aliases..."
  ollama cp "$model" claude-sonnet-4-6 2>/dev/null && success "claude-sonnet-4-6 → $model"
  ollama cp "$model" claude-haiku-4-5  2>/dev/null && success "claude-haiku-4-5  → $model"
  # If 14b is available, use it for opus
  if ollama list 2>/dev/null | grep -q "qwen3:14b" && [[ "$model" != "qwen3:14b" ]]; then
    ollama cp qwen3:14b claude-opus-4-5 2>/dev/null && success "claude-opus-4-5   → qwen3:14b"
  else
    ollama cp "$model" claude-opus-4-5 2>/dev/null && success "claude-opus-4-5   → $model"
  fi
}

# ── write env helper ──────────────────────────────────────────────────────────
write_env_file() {
  cat > .env.qwen-local <<'EOF'
# Source this to use Claude Code with your local Qwen model:
#   source .env.qwen-local && claude
export ANTHROPIC_BASE_URL=http://localhost:11434
export ANTHROPIC_AUTH_TOKEN=ollama
EOF
  success "Wrote .env.qwen-local"
}

# ── chat UI server ────────────────────────────────────────────────────────────
start_chat_ui() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  local html="$script_dir/chat.html"

  if [[ ! -f "$html" ]]; then
    warn "chat.html not found — skipping UI server"
    return
  fi

  local port=8080
  while lsof -i ":$port" &>/dev/null 2>&1; do (( port++ )); done

  cd "$script_dir"
  if command -v python3 &>/dev/null; then
    python3 -m http.server "$port" --bind 127.0.0.1 &>/tmp/qwen-ui.log &
  elif command -v python &>/dev/null; then
    python -m SimpleHTTPServer "$port" &>/tmp/qwen-ui.log &
  else
    warn "python3 not found — can't auto-start UI. Open chat.html directly in your browser."
    return
  fi
  success "Chat UI running at ${C}http://localhost:${port}${N}"
}

# ── print Claude Code instructions ───────────────────────────────────────────
print_claude_instructions() {
  local model=$1
  echo
  echo -e "${W}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${N}"
  echo -e "${W}  Using Claude Code with your local model${N}"
  echo -e "${W}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${N}"
  echo
  echo -e "  Launch a local Claude Code session:"
  echo
  echo -e "    ${C}source .env.qwen-local && claude${N}"
  echo
  echo -e "  Or inline:"
  echo
  echo -e "    ${C}ANTHROPIC_BASE_URL=http://localhost:11434 \\${N}"
  echo -e "    ${C}ANTHROPIC_AUTH_TOKEN=ollama \\${N}"
  echo -e "    ${C}claude${N}"
  echo
  echo -e "  ${Y}Note:${N} this routes Claude Code through your local"
  echo -e "  ${Y}$model${N} — no API key, no cloud, no cost."
  echo -e "  Agentic tasks work best with 14b+ models."
  echo
}

# ── main ──────────────────────────────────────────────────────────────────────
main() {
  banner

  local platform
  platform=$(detect_platform)
  info "Platform: ${W}$platform${N}"

  if [[ "$platform" == "unknown" ]]; then
    warn "Unrecognised platform. Assuming Linux — some steps may fail."
    platform="linux"
  fi

  # Hardware
  local vram ram gpu_name
  vram=$(detect_vram_gb)
  ram=$(detect_ram_gb)
  gpu_name=$(detect_gpu_name)

  info "GPU:  ${W}${gpu_name}${N}"
  info "VRAM: ${W}${vram} GB${N}  RAM: ${W}${ram} GB${N}"
  echo

  # Model recommendation
  local recommended
  recommended=$(pick_model "$vram" "$platform" "$ram")
  local size_gb
  size_gb=$(model_size_gb "$recommended")

  echo -e "  ${G}Recommended model:${N} ${W}${recommended}${N} (~${size_gb} GB download)"
  echo
  echo -e "  Available models (choose or press Enter to accept):"
  echo -e "    ${C}1${N}) qwen3:1.7b  — 1.4 GB  ultrafast, limited reasoning"
  echo -e "    ${C}2${N}) qwen3:4b    — 2.6 GB  good for low-VRAM / CPU"
  echo -e "    ${C}3${N}) qwen3:8b    — 5.2 GB  strong all-rounder ← sweet spot for 8 GB VRAM"
  echo -e "    ${C}4${N}) qwen3:14b   — 9.3 GB  noticeably smarter, needs ~10 GB VRAM"
  echo -e "    ${C}5${N}) qwen3:32b   — 20.5 GB best quality, needs 24 GB VRAM"
  echo
  ask "Pick [1-5] or Enter for recommended (${recommended}): "
  read -r choice

  local model
  case "$choice" in
    1) model="qwen3:1.7b" ;;
    2) model="qwen3:4b"   ;;
    3) model="qwen3:8b"   ;;
    4) model="qwen3:14b"  ;;
    5) model="qwen3:32b"  ;;
    *) model="$recommended" ;;
  esac

  echo
  info "Selected: ${W}${model}${N} (~$(model_size_gb "$model") GB)"
  ask "Download and set up ${model}? [Y/n]: "
  read -r confirm
  [[ "${confirm,,}" == "n" ]] && echo "Aborted." && exit 0

  echo

  # Install + start Ollama
  install_ollama "$platform"
  ensure_ollama_running

  # Pull model
  info "Pulling ${model} (this may take a few minutes)..."
  ollama pull "$model"
  success "Model ready"

  # Claude Code aliases
  setup_claude_aliases "$model"

  # .env file
  write_env_file

  # Chat UI
  start_chat_ui

  # Done
  echo
  echo -e "${G}  ✓ All done!${N}"
  print_claude_instructions "$model"
}

main "$@"
