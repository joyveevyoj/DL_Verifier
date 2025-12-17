# DL_Verifier

Test-time scaling experiment for LLM verification using Qwen3-0.6B on GSM8K dataset.

## Environment Setup

### Using Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate llm-verifier
```

### Using pip

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirement.txt
```

### For CUDA (>= 12.1) (Recommended)
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

### For MacOS (Not Recommended)
```bash
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0
```

<!-- ```bash
pip install -e .
``` -->
