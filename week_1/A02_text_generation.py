# GPT-2 ဖြင့် robot command ဆက် generate လုပ်ကြည့်ပါ
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# TODO: "Move the robot to" ဆိုသော prompt ကနေ sentence ဆက်ထုတ်ကြည့်ပါ

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

input_text = "Move the robot to"
inputs = tokenizer(input_text, return_tensors="pt")

output = model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=10, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)