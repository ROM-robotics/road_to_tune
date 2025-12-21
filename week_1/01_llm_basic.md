# LLM Basics: Encoder / Decoder / Transformer

## မိတ်ဆက်

Large Language Models (LLM) များသည် Transformer architecture အပေါ်အခြေခံထားသော AI models များဖြစ်ပါသည်။ ဤသင်ခန်းစာတွင် Encoder, Decoder နှင့် Transformer တို့၏ အခြေခံသဘောတရားများကို လက်တွေ့ကျကျ လေ့လာမည်ဖြစ်ပါသည်။

## Transformer Architecture အခြေခံများ

### ၁. Encoder (ကုဒ်ဝှက်သူ)

**သဘောတရား:**
Encoder သည် input text ကို နားလည်ပြီး အဓိပ္ပာယ်ရှိသော representation (vector) အဖြစ် ပြောင်းလဲပေးသည်။

**အလုပ်လုပ်ပုံ:**
```
Input Text → Tokenization → Embedding → Self-Attention → Feed Forward → Output Representation
```

**လက်တွေ့ဥပမာ:**
```python
# BERT သည် Encoder-only model ဖြစ်သည်
from transformers import BertTokenizer, BertModel

# Model နှင့် Tokenizer load လုပ်ခြင်း
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# မြန်မာစာ input (English example)
text = "The robot is moving forward"
inputs = tokenizer(text, return_tensors="pt")

# Encoder က representation ထုတ်ပေးခြင်း
outputs = model(**inputs)
last_hidden_state = outputs.last_hidden_state  # Shape: [1, seq_len, 768]

print(f"Input text: {text}")
print(f"Encoded representation shape: {last_hidden_state.shape}")
```

**Encoder အသုံးပြုရာများ:**
- Text Classification (စာသား အမျိုးအစား ခွဲခြားခြင်း)
- Named Entity Recognition (အမည်သတ်မှတ်ခြင်း)
- Sentiment Analysis (စိတ်ခံစားချက် ခွဲခြမ်းစိတ်ဖြာခြင်း)

---

### ၂. Decoder (ကုဒ်ဖြေသူ)

**သဘောတရား:**
Decoder သည် စာသားများကို တစ်လုံးချင်းစီ ထုတ်ပေးနိုင်သော architecture ဖြစ်သည်။ ယခင် generated words များကို အခြေခံ၍ နောက်ထပ် word ကို predict လုပ်သည်။

**အလုပ်လုပ်ပုံ:**
```
Input Prompt → Embedding → Masked Self-Attention → Feed Forward → Next Token Prediction
```

**လက်တွေ့ဥပမာ:**
```python
# GPT-2 သည် Decoder-only model ဖြစ်သည်
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Model load လုပ်ခြင်း
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Text generation လုပ်ခြင်း
prompt = "The autonomous robot can"
inputs = tokenizer(prompt, return_tensors="pt")

# Decoder က စာသားဆက် ထုတ်ပေးခြင်း
outputs = model.generate(
    inputs.input_ids,
    attention_mask=inputs.get("attention_mask"),  # explicit attention mask
    max_length=50,
    num_return_sequences=1,
    do_sample=True,            # sampling mode ကို ဖွင့်ထားရန် (temperature 有効化)
    temperature=0.7,          # 이제 유효합니다
    pad_token_id=tokenizer.eos_token_id
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Prompt: {prompt}")
print(f"Generated: {generated_text}")
```

**Decoder အသုံးပြုရာများ:**
- Text Generation (စာသား ဖန်တီးခြင်း)
- Story Writing (ဇာတ်လမ်း ရေးသားခြင်း)
- Code Generation (ကုဒ် ထုတ်ပေးခြင်း)

---

### ၃. Encoder-Decoder (ပေါင်းစပ်မော်ဒယ်)

**သဘောတရား:**
Encoder-Decoder model သည် Encoder ၏ နားလည်မှုစွမ်းရည်နှင့် Decoder ၏ ထုတ်ပေးနိုင်မှုစွမ်းရည် နှစ်ခုလုံးကို ပေါင်းစပ်ထားသည်။

**အလုပ်လုပ်ပုံ:**
```
Input → Encoder → Context Vector → Decoder → Output
```

**လက်တွေ့ဥပမာ:**
```python
# T5 သည် Encoder-Decoder model ဖြစ်သည်
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Model load လုပ်ခြင်း
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# Translation task
input_text = "translate English to German: The robot moves to the goal position"
inputs = tokenizer(input_text, return_tensors="pt")

# Encoder-Decoder process
outputs = model.generate(inputs.input_ids, max_length=50)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Input: {input_text}")
print(f"Output: {result}")
```

**Encoder-Decoder အသုံးပြုရာများ:**
- Machine Translation (ဘာသာပြန်ခြင်း)
- Summarization (အကျဉ်းချုပ်ခြင်း)
- Question Answering (မေးခွန်းဖြေဆိုခြင်း)

---

## Self-Attention Mechanism

**သဘောတရား:**
Self-Attention သည် input sequence ရှိ word တစ်ခုချင်းစီက အခြား words များနှင့် ဘယ်လောက် ဆက်စပ်နေသည်ကို တွက်ချက်ပေးသည်။

**လက်တွေ့ကြည့်ရှုခြင်း:**
```python
import torch
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)

# ဝါကျ ထည့်ခြင်း
text = "The robot navigates to the charging station"
inputs = tokenizer(text, return_tensors="pt")

# Attention weights ထုတ်ကြည့်ခြင်း
outputs = model(**inputs)
attentions = outputs.attentions  # Tuple of attention weights from each layer

# First layer, first head ၏ attention ကြည့်ခြင်း
first_layer_attention = attentions[0][0, 0].detach()
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

print("Tokens:", tokens)
print("Attention matrix shape:", first_layer_attention.shape)
print("\nAttention scores (first token attending to others):")
for i, token in enumerate(tokens):
    print(f"{tokens[0]} → {token}: {first_layer_attention[0, i]:.4f}")
```

---

## Model Sizes နှင့် Parameters

**နားလည်သင့်သော အချက်များ:**

| Model Type | Example | Parameters | Use Case |
|------------|---------|------------|----------|
| Small | BERT-base | 110M | Classification, NER |
| Medium | GPT-2 | 355M-774M | Text generation |
| Large | GPT-3 | 175B | Complex reasoning |
| Very Large | GPT-4 | Unknown (>1T) | Multi-modal tasks |

**ROS2/Robotics အတွက် ရွေးချယ်ပုံ:**
- Real-time response လိုအပ်ပါက: Small models (BERT, DistilBERT)
- Complex reasoning လိုအပ်ပါက: Medium to Large models
- Edge devices များတွင်: Quantized or distilled models

---

## လေ့ကျင့်ခန်း

### Exercise 1: Encoder နားလည်ခြင်း
```python
# BERT ဖြင့် sentence embedding ထုတ်ကြည့်ပါ
from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

sentences = [
    "The robot is moving forward",
    "The robot is stationary",
    "The autonomous vehicle navigates"
]

# TODO: Sentence embeddings ထုတ်ပြီး similarity တွက်ကြည့်ပါ
# check A01_sentense_similarity_cpu.py
```

### Exercise 2: Decoder text generation
```python
# GPT-2 ဖြင့် robot command ဆက် generate လုပ်ကြည့်ပါ
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# TODO: "Move the robot to" ဆိုသော prompt ကနေ sentence ဆက်ထုတ်ကြည့်ပါ
# check A02_text_generation.py
```

### Exercise 3: Encoder-Decoder translation
```python
# T5 ဖြင့် robot command များကို အခြား format သို့ ပြောင်းကြည့်ပါ
# TODO: Natural language command ကို ROS2 command format သို့ ပြောင်းကြည့်ပါ
# check A03_encode_decode.py
```

---

## အနှစ်ချုပ်

- **Encoder**: Input ကို နားလည်ပြီး representation ထုတ်ပေးသည်
- **Decoder**: စာသားကို တစ်လုံးချင်းစီ ထုတ်ပေးသည်
- **Encoder-Decoder**: နှစ်ခု ပေါင်းစပ်ပြီး translation/summarization လုပ်သည်
- **Self-Attention**: Words များအကြား ဆက်စပ်မှုကို တွက်ချက်သည်

နောက်သင်ခန်းစာတွင် Prompt Engineering နည်းလမ်းများကို ဆက်လက်လေ့လာပါမည်။
