# T5 ဖြင့် robot command များကို အခြား format သို့ ပြောင်းကြည့်ပါ
from transformers import T5Tokenizer, T5ForConditionalGeneration

# TODO: Natural language command ကို ROS2 command format သို့ ပြောင်းကြည့်ပါ

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Note: Without fine-tuning, T5 won't know actual ROS2 commands. 
# This is just the code structure for inference.
input_text = "translate English to ROS2: Move the robot forward"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

outputs = model.generate(input_ids, max_length=50)
print("Input:", input_text)
print("Output:", tokenizer.decode(outputs[0], skip_special_tokens=True))
