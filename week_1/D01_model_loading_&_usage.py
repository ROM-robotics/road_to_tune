from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch

model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

# ၁.၁ Tokenizer Load
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

# Text tokenize
text = "The robot moves forward 1 meter"
tokens = tokenizer(text, return_tensors="pt")

print("Input IDs:", tokens.input_ids)
print("Tokens:", tokenizer.convert_ids_to_tokens(tokens.input_ids[0]))

# ၁.၂ Base Model Load (hidden states / features)
model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True
)

with torch.no_grad():
    outputs = model(**tokens)

print("Output shape:", outputs.last_hidden_state.shape)

# ၁.၃ Generation Model
gen_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True
)

output_ids = gen_model.generate(
    **tokens,
    max_length=50,
    temperature=0.3,          # better for instructions
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

generated_text = tokenizer.decode(
    output_ids[0],
    skip_special_tokens=True
)

print("Generated:", generated_text)
