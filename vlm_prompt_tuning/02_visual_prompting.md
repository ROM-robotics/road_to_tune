# Visual Prompt (Image + Text) Conditioning

Visual Prompting ဆိုသည်မှာ VLM တစ်ခုကို ပုံရိပ် (Image) နှင့် စာသား (Text) ညွှန်ကြားချက်များ ပေါင်းစပ်ပေးပို့ပြီး လိုချင်သော အဖြေကို ရယူခြင်းဖြစ်သည်။

## 1. Basic Structure
VLM များအတွက် prompt structure သည် ပုံမှန်အားဖြင့် အောက်ပါအတိုင်း ဖြစ်သည် -
```text
<Image_Placeholder> + <System_Prompt> + <User_Instruction>
```

### Example (Python Code with Transformers)
```python
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image

# Load Model
model_id = "llava-hf/llava-1.5-7b-hf"
model = LlavaForConditionalGeneration.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

# Prepare Inputs
image = Image.open("robot_view.jpg")
prompt = "USER: <image>\nWhat objects are in front of the robot?\nASSISTANT:"

inputs = processor(text=prompt, images=image, return_tensors="pt")

# Generate
generate_ids = model.generate(**inputs, max_new_tokens=50)
print(processor.batch_decode(generate_ids, skip_special_tokens=True)[0])
```

## 2. Conditioning Techniques

### A. Direct Visual Question Answering (VQA)
ပုံထဲရှိ အကြောင်းအရာကို တိုက်ရိုက်မေးမြန်းခြင်း။
- **Prompt:** "Is the door open or closed?"
- **Use Case:** Robot navigation အတွက် အခြေအနေစစ်ဆေးခြင်း။

### B. Visual Grounding / Referring Expression
ပုံထဲရှိ အရာဝတ္ထုတစ်ခု၏ တည်နေရာကို မေးမြန်းခြင်း။
- **Prompt:** "Where is the red ball? Give me the bounding box coordinates."
- **Output:** "[0.2, 0.4, 0.3, 0.5]" (Normalized coordinates)

### C. Interleaved Image-Text Prompting
ပုံများစွာနှင့် စာသားများကို ရောနှောအသုံးပြုခြင်း (Few-shot learning အတွက် အသုံးဝင်သည်)။
- **Prompt:** 
  - Image 1 (Apple) + Text: "This is a fruit."
  - Image 2 (Car) + Text: "This is a vehicle."
  - Image 3 (Banana) + Text: "This is a..."
- **Output:** "fruit."

## 3. Prompt Engineering for Robotics
Robot များအတွက် တိကျသော အဖြေရရန် prompt ကို ရှင်းလင်းစွာ ရေးသားရမည်။

**Bad Prompt:**
"What do you see?" (အဖြေက အရမ်းကျယ်ပြန့်နိုင်သည်)

**Good Prompt (Conditioning):**
"Analyze the image for navigation hazards. List any obstacles within 2 meters in JSON format."

**Example Output:**
```json
{
  "obstacles": [
    {"object": "chair", "distance": "1.5m", "position": "left"},
    {"object": "box", "distance": "0.5m", "position": "center"}
  ]
}
```

## Summary
Visual Prompting သည် model အား ပုံကိုကြည့်ရှုနည်းနှင့် မည်သည့်အချက်အလက်ကို ဆွဲထုတ်ရမည်ကို လမ်းညွှန်ပေးခြင်းဖြစ်သည်။ Text prompt ကောင်းမွန်လေ၊ VLM ၏ စွမ်းဆောင်ရည် မြင့်မားလေဖြစ်သည်။
