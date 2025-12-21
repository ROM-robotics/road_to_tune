from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# အသုံးပြုမည့် Model ID ကို သတ်မှတ်ခြင်း (Qwen 2.5 3B Instruct version)
model_id = "Qwen/Qwen2.5-3B-Instruct"

# Tokenizer နှင့် Model ကို Load လုပ်ခြင်း
# torch_dtype=torch.float16 သည် memory သက်သာစေရန် half-precision ကို သုံးထားခြင်းဖြစ်သည်
# device_map="auto" သည် GPU ရှိလျှင် GPU ပေါ်သို့ အလိုအလျောက် တင်ပေးမည်
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

messages = [
    {
        "role": "system", 
        "content": "You are a ROS2 tutor. Explain simply."
    },
    {
        "role": "user", 
        "content": "what is ROS2 node."
    }
]

# Chat template အသုံးပြု၍ prompt format ပြောင်းခြင်း (OpenAI style မှ HF style သို့)
# tokenize=False ထားခြင်းသည် raw string အနေဖြင့်သာ လိုချင်သောကြောင့်ဖြစ်သည်
input_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# စာသားများကို Model နားလည်သော tensor များအဖြစ် ပြောင်းလဲခြင်း (Tokenization)
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

# အဖြေထုတ်ခြင်း (Generation)
# max_new_tokens: အများဆုံး ထုတ်ပေးမည့် စာလုံးအရေအတွက်
# temperature: အဖြေ၏ ကွဲပြားနိုင်မှု (0.3 သည် တည်ငြိမ်သော အဖြေကို ပေးသည်)
outputs = model.generate(
    **inputs,
    max_new_tokens=300,
    temperature=0.3
)

# ရလာသော output tensor များကို လူနားလည်သော စာသားအဖြစ် ပြန်ပြောင်းခြင်း (Decoding)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
