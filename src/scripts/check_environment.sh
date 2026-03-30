#!/bin/bash

# ========================================
# Environment Check Script
# Verify all requirements before training
# ========================================

echo "================================================"
echo "TON UVB Environment Check"
echo "================================================"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track overall status
ALL_CHECKS_PASSED=true

# Function to print check result
print_check() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $2"
    else
        echo -e "${RED}✗${NC} $2"
        ALL_CHECKS_PASSED=false
    fi
}

# Function to print warning
print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

echo "1. Checking Python Environment..."
echo "-----------------------------------"

# Check Python version
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 11 ]; then
        print_check 0 "Python $PYTHON_VERSION (>= 3.11)"
    else
        print_check 1 "Python version: $PYTHON_VERSION (need >= 3.11)"
    fi
else
    print_check 1 "Python not found"
fi

# Check conda
if command -v conda &> /dev/null; then
    print_check 0 "Conda installed"
else
    print_warning "Conda not found (optional, but recommended)"
fi

echo ""
echo "2. Checking GPU Setup..."
echo "-----------------------------------"

# Check CUDA
if command -v nvidia-smi &> /dev/null; then
    print_check 0 "NVIDIA drivers installed"
    
    # Get GPU count
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "   Found $GPU_COUNT GPU(s):"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | while read line; do
        echo "   - GPU $line"
    done
    
    # Check VRAM
    MIN_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | head -1)
    if [ "$MIN_VRAM" -ge 20000 ]; then
        print_check 0 "Minimum VRAM: ${MIN_VRAM}MB (>= 20GB)"
    else
        print_check 1 "Minimum VRAM: ${MIN_VRAM}MB (need >= 20GB, recommend 24GB+)"
    fi
else
    print_check 1 "NVIDIA drivers not found (nvidia-smi not available)"
fi

# Check CUDA toolkit (nvcc)
if command -v nvcc &> /dev/null; then
    NVCC_VERSION=$(nvcc --version | awk -F'release ' '/release/{print $2}' | awk -F',' '{print $1}')
    if [[ "${NVCC_VERSION}" == "12.4" ]]; then
        print_check 0 "nvcc release ${NVCC_VERSION} (expected 12.4)"
    else
        print_warning "nvcc release ${NVCC_VERSION} (runbook target: 12.4)"
    fi
else
    print_warning "nvcc not found (CUDA toolkit check skipped)"
fi

# Check CUDA availability in Python
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null)
    print_check 0 "PyTorch CUDA available (version: $CUDA_VERSION)"
else
    print_check 1 "PyTorch CUDA not available"
fi

echo ""
echo "3. Checking Python Packages..."
echo "-----------------------------------"

# Check critical packages
check_package() {
    if python -c "import $1" 2>/dev/null; then
        VERSION=$(python -c "import $1; print($1.__version__ if hasattr($1, '__version__') else 'unknown')" 2>/dev/null)
        print_check 0 "$1 ($VERSION)"
        return 0
    else
        print_check 1 "$1 not installed"
        return 1
    fi
}

check_package "torch"
check_package "transformers"
check_package "peft"
check_package "trl"
check_package "datasets"
check_package "yt_dlp"
check_package "vllm"
check_package "deepspeed"
check_package "accelerate"
check_package "flash_attn"

echo ""
echo "Version compatibility checks (runbook target):"
python - <<'PY' 2>/dev/null || true
import importlib

targets = {
    "torch": "2.5.1+cu124",
    "transformers": None,  # pinned by git revision in setup
    "peft": "0.14.0",
    "trl": "0.14.0",
    "deepspeed": "0.15.4",
    "vllm": "0.7.2",
    "flash_attn": "2.6.3",
}

for pkg, target in targets.items():
    try:
        mod = importlib.import_module(pkg)
        got = getattr(mod, "__version__", "unknown")
        if target is None:
            print(f"  - {pkg}: {got}")
        elif got == target:
            print(f"  - {pkg}: {got} (OK)")
        else:
            print(f"  - {pkg}: {got} (target {target})")
    except Exception:
        print(f"  - {pkg}: not installed")
PY

# Check optional packages
echo ""
echo "Optional packages:"
if python -c "import wandb" 2>/dev/null; then
    print_check 0 "wandb (recommended for monitoring)"
else
    print_warning "wandb not installed (optional, but recommended)"
    echo "   Install with: pip install wandb"
fi

echo ""
echo "4. Checking Disk Space..."
echo "-----------------------------------"

# Check available disk space (portable across Linux/macOS)
DISK_AVAILABLE_HUMAN=$(df -h . | awk 'NR==2 {print $4}')
DISK_AVAILABLE_KB=$(df -Pk . | awk 'NR==2 {print $4}')

if [[ -n "${DISK_AVAILABLE_KB}" && "${DISK_AVAILABLE_KB}" =~ ^[0-9]+$ ]]; then
    DISK_AVAILABLE_GB=$((DISK_AVAILABLE_KB / 1024 / 1024))
else
    DISK_AVAILABLE_GB=""
fi

echo "   Available disk space: ${DISK_AVAILABLE_HUMAN}"
if [[ -n "${DISK_AVAILABLE_GB}" ]]; then
    if [ "${DISK_AVAILABLE_GB}" -ge 100 ]; then
        print_check 0 "Disk space: ${DISK_AVAILABLE_GB}GB (>= 100GB)"
    else
        print_warning "Disk space: ${DISK_AVAILABLE_GB}GB (recommend >= 100GB)"
        echo "   Needed for: model (~30GB) + data (~10GB) + checkpoints (~60GB)"
    fi
else
    print_warning "Unable to parse disk space in GB"
fi

echo ""
echo "5. Checking Repository Structure..."
echo "-----------------------------------"

# Check critical directories and files
check_path() {
    if [ -e "$1" ]; then
        print_check 0 "$1"
        return 0
    else
        print_check 1 "$1 not found"
        return 1
    fi
}

check_path "src/r1-v/src/open_r1/grpo_uvb.py"
check_path "src/scripts/run_grpo_uvb_answer_only.sh"
check_path "src/scripts/prepare_all_grpo_data.sh"
check_path "src/scripts/prepare_uvb_full_split_local_videos.sh"
check_path "src/scripts/prepare_uvb_grpo_data.sh"
check_path "src/scripts/prepare_videommmu_grpo_data.sh"
check_path "src/scripts/prepare_mmvu_grpo_data.sh"
check_path "src/r1-v/configs/zero1_no_optimizer.json"

MERGED_MODEL_DIR="${MERGED_MODEL_DIR:-sft/outputs/qwen25vl3b_lora_merged_from_sft40}"
echo ""
echo "Merged model metadata check (${MERGED_MODEL_DIR})..."
if [ -d "${MERGED_MODEL_DIR}" ]; then
    check_path "${MERGED_MODEL_DIR}/config.json"
    check_path "${MERGED_MODEL_DIR}/tokenizer.json"
    check_path "${MERGED_MODEL_DIR}/tokenizer_config.json"
    check_path "${MERGED_MODEL_DIR}/preprocessor_config.json"
else
    print_warning "Merged model dir not found: ${MERGED_MODEL_DIR}"
fi

echo ""
echo "6. Checking Network Access..."
echo "-----------------------------------"

# Check HuggingFace access
if curl -s --head https://huggingface.co | head -n 1 | grep "HTTP" > /dev/null; then
    print_check 0 "HuggingFace accessible"
else
    print_warning "Cannot reach huggingface.co (check network)"
fi

echo ""
echo "================================================"
if [ "$ALL_CHECKS_PASSED" = true ]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    echo "You're ready to start training."
    echo ""
    echo "Next steps:"
    echo "  1. Prepare all GRPO datasets: bash src/scripts/prepare_all_grpo_data.sh"
    echo "  2. Start training/eval: bash src/scripts/run_grpo_uvb_answer_only.sh"
else
    echo -e "${RED}✗ Some checks failed${NC}"
    echo "Please fix the issues above before training."
    echo ""
    echo "Common fixes:"
    echo "  - Install dependencies: bash setup.sh"
    echo "  - Install vllm: pip install vllm==0.7.2"
    echo "  - Check GPU drivers: nvidia-smi"
fi
echo "================================================"

exit $([ "$ALL_CHECKS_PASSED" = true ] && echo 0 || echo 1)
