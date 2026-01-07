from transformers import AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig
import torch.nn as nn
import torch


def build_bce_model(model_name, lora_r, lora_alpha, lora_dropout, pad_token_id, config):
    model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=1, 
    device_map=None,  # Use .to(device) instead for manual placement
    torch_dtype=torch.bfloat16
    )
  
    model.config.pad_token_id = pad_token_id

    # Apply LoRA
    peft_config = LoraConfig(
        task_type="SEQ_CLS",
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout, #0.05
        target_modules=["q_proj", "v_proj"],
        # use_dora=True #DoRA option, maybe for bigger model
    )
    model = get_peft_model(model, peft_config)
    return model