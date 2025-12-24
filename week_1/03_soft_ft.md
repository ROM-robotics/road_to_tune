# Soft Fine-Tuning: Prompt Tuning & P-Tuning

## မိတ်ဆက်

Soft Fine-Tuning (သို့) Soft Prompting သည် model ၏ parameters များကို update မလုပ်ဘဲ trainable "soft prompts" များ ထည့်သွင်း၍ model ကို specific task အတွက် adapt လုပ်ခြင်း ဖြစ်သည်။ ဤနည်းလမ်းသည် parameter-efficient ဖြစ်ပြီး multiple tasks များအတွက် အသုံးပြုနိုင်သည်။

## Hard Prompts vs Soft Prompts

### Hard Prompts (ရိုးရာ Prompting)
```
Input: "Translate to Myanmar: Hello" → Model → Output
```
- လူသားဖတ်နိုင်သော စာသားများ အသုံးပြုသည်
- Manual ရေးသားရသည်
- Fixed words များ

### Soft Prompts (Learnable Embeddings)
```
Input: [learnable_emb_1][learnable_emb_2]..."Translate: Hello" → Model → Output
```
- Continuous vector space တွင် ရှိသည်
- Training ဖြင့် optimize လုပ်နိုင်သည်
- Task-specific ဖြစ်သည်

---

## ၁. Prompt Tuning

### သဘောတရား

Prompt Tuning သည် model input ရှေ့တွင် learnable continuous vectors (soft prompts) များ ထည့်ပေးပြီး အဲဒီ vectors များကိုသာ train လုပ်သည်။ Base model parameters များသည် frozen (မပြောင်းလဲ) ဖြစ်နေသည်။

### Architecture

```
┌─────────────────────────────────────────┐
│  Soft Prompt (Learnable)                │
│  [emb_1, emb_2, ..., emb_n]            │
├─────────────────────────────────────────┤
│  Input Text (Fixed)                     │
│  "Move robot to charging station"      │
├─────────────────────────────────────────┤
│  Pretrained LLM (Frozen)               │
│  ❄️  No parameter updates               │
├─────────────────────────────────────────┤
│  Output                                 │
│  "ros2 action send_goal ..."           │
└─────────────────────────────────────────┘
```



### Prompt Tuning Training

```bash
# check C02

```
---

## ၂. P-Tuning

### သဘောတရား

P-Tuning သည် Prompt Tuning ကဲ့သို့ပင် ဖြစ်သော်လည်း soft prompts များကို input ၏ အမျိုးမျိုးသော positions များတွင် ထည့်နိုင်သည်။ အခြား continuous prompts များ အကြား discrete tokens များ ရောနှောထည့်နိုင်သည်။

### P-Tuning Architecture

```
Input: [soft_1] "Navigate" [soft_2] "to" [soft_3] "goal"
                    ↓
              Frozen LLM
                    ↓
               Output
```

### P-Tuning Implementation

```bash
# check C02
```

---

## Prompt Tuning vs P-Tuning Comparison

| Feature | Prompt Tuning | P-Tuning v1 | P-Tuning v2 |
|---------|---------------|-------------|-------------|
| Soft prompt location | Input prefix only | Interleaved | Every layer |
| Encoder | None | LSTM | None |
| Trainable params | Very few | Few | Moderate |
| Performance | Good | Better | Best |
| Complexity | Simple | Moderate | Complex |


---


## လေ့ကျင့်ခန်း

### Exercise 1: Implement Basic Prompt Tuning
```python
# TODO: GPT-2 အတွက် prompt tuning model တစ်ခု implement လုပ်ပါ
# Requirements:
# - 10 soft prompt tokens
# - Train with 5 ROS2 examples
# - Loss curve plot လုပ်ပါ

class SimplePromptTuning(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        # TODO: Implement
        pass
```

### Exercise 2: Compare Prompt Lengths
```python
# TODO: Soft prompt length (5, 10, 20, 50) များအတွက် performance နှိုင်းယှဉ်ပါ
# Which length gives best results for ROS2 command generation?

results = {}
for n_tokens in [5, 10, 20, 50]:
    # TODO: Train and evaluate
    pass
```

### Exercise 3: Multi-Task Prompt Tuning
```python
# TODO: ROS2 tasks သုံးခု အတွက် soft prompts သုံးခု train လုပ်ပါ
# Task 1: Command generation
# Task 2: Error explanation
# Task 3: Code review

tasks = {
    "command_gen": [...],
    "error_explain": [...],
    "code_review": [...]
}

# TODO: Implement multi-task learning
```

---

## အနှစ်ချုပ်

- **Soft Fine-Tuning**: Learnable continuous prompts များ သုံး၍ model adapt လုပ်ခြင်း
- **Prompt Tuning**: Input prefix တွင် soft prompts ထည့်ခြင်း
- **P-Tuning**: Interleaved positions များတွင် soft prompts ထည့်ခြင်း
- **Advantages**: Parameter efficient, memory efficient, multi-task learning ပံ့ပိုးမှု

**Key Insight**: Model ၏ billions of parameters များကို ပြောင်းစရာမလို။ Thousands of soft prompt parameters များ train လုပ်ရုံဖြင့် task-specific performance ရရှိနိုင်သည်။

နောက်သင်ခန်းစာတွင် HuggingFace Transformers & PEFT library များကို အသေးစိတ် လေ့လာပါမည်။
