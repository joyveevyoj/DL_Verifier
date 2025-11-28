# DL_Verifier

Test-time scaling experiment for LLM verification using Qwen3-0.6B on GSM8K dataset.

## Environment Setup

### Using Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate llm-verifier
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0
```

<!-- ### Using pip

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirement.txt
``` -->

## Finetune

### Unzip Training Data

```bash
unzip verifier_dataset_train.zip -d ./data
```
