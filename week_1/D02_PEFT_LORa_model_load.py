from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


# -------------------------------------------------------------------
# Load tokenizer (MUST match training)
# -------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-Coder-1.5B-Instruct"
)
tokenizer.pad_token = tokenizer.eos_token


# -------------------------------------------------------------------
# Load base model
# -------------------------------------------------------------------
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    device_map="auto"
)


# -------------------------------------------------------------------
# Load LoRA weights
# -------------------------------------------------------------------
peft_model = PeftModel.from_pretrained(
    base_model,
    "/home/mr_robot/Downloads/ros2_command_model_final"
)

peft_model.eval()  # ✅ important for inference


# -------------------------------------------------------------------
# Inference
# -------------------------------------------------------------------
prompt = """### Instruction:
Move robot forward 3 meters

### Command:
"""

inputs = tokenizer(prompt, return_tensors="pt").to(peft_model.device)

outputs = peft_model.generate(
    **inputs,
    max_new_tokens=100,   # ✅ correct
    temperature=0.7,
    do_sample=True,
    eos_token_id=tokenizer.eos_token_id,
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
