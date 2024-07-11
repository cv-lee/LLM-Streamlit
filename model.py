import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = './ckpt/Qwen2-7B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
    )
model.eval()

