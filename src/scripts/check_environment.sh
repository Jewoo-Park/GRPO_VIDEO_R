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
check_package "datasets"
check_package "vllm"
check_package "deepspeed"
check_package "accelerate"

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

# Check available disk space
DISK_AVAILABLE=$(df -h . | awk 'NR==2 {print $4}')
DISK_AVAILABLE_GB=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')

echo "   Available disk space: $DISK_AVAILABLE"
if [ "$DISK_AVAILABLE_GB" -ge 100 ]; then
    print_check 0 "Disk space: ${DISK_AVAILABLE_GB}GB (>= 100GB)"
else
    print_warning "Disk space: ${DISK_AVAILABLE_GB}GB (recommend >= 100GB)"
    echo "   Needed for: model (~30GB) + data (~10GB) + checkpoints (~60GB)"
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
check_path "src/scripts/prepare_uvb_dataset.sh"
check_path "src/scripts/prepare_uvb_40_split_download_frames.sh"
check_path "src/scripts/prepare_uvb_grpo_data.sh"
check_path "src/r1-v/configs/zero1_no_optimizer.json"

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
    echo "  1. Prepare UVB metadata: bash src/scripts/prepare_uvb_dataset.sh"
    echo "  2. Sample/split/download/extract: bash src/scripts/prepare_uvb_40_split_download_frames.sh"
    echo "  3. Build GRPO JSONL: bash src/scripts/prepare_uvb_grpo_data.sh"
    echo "  4. Start training: bash src/scripts/run_grpo_uvb_answer_only.sh"
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
