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

### လက်တွေ့အကောင်အထည်ဖော်မှု

```bash
# check C02

```

### Prompt Tuning Training

```python

from torch.optim import AdamW
from torch.utils.data import DataLoader

# ROS2 command generation dataset
train_data = [
    {
        "input": "Move forward 2 meters",
        "output": "ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \"{linear: {x: 0.5}}\""
    },
    {
        "input": "Turn left 90 degrees",
        "output": "ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \"{angular: {z: 1.57}}\""
    },
    # More examples...
]

def train_prompt_tuning(model, train_data, epochs=50):
    device = next(model.parameters()).device
    optimizer = AdamW([model.soft_prompt], lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    for epoch in range(epochs):
        total_loss = 0.0

        for item in train_data:
            input_text = f"""### Instruction:
            Convert the following command into a ROS2 command.
            
            ### Input:
            {item['input']}
            
            ### Output:
            """
            target_text = item["output"]

            # Tokenize
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
            target_ids = tokenizer(target_text, return_tensors="pt").input_ids.to(device)

            # Concatenate input + target (token ids used to construct embeddings)
            full_input_ids = torch.cat([input_ids, target_ids], dim=1)

            # Attention mask for tokens (will be expanded inside forward with prompt prefix)
            attention_mask = torch.ones_like(full_input_ids).to(device)

            # Forward
            outputs = model(full_input_ids, attention_mask)
            logits = outputs.logits   # (batch, seq_len_with_prompt, vocab)

            # Build labels that account for the soft prompt prefix so shapes match logits
            batch_size = full_input_ids.size(0)
            n_prompt = model.n_prompt_tokens
            seq_len_tokens = full_input_ids.size(1)

            # labels for the token positions (no loss on input tokens)
            token_labels = torch.full_like(full_input_ids, -100).to(device)
            token_labels[:, -target_ids.size(1):] = target_ids

            # prepend -100 labels for the soft prompt positions
            prompt_labels = torch.full((batch_size, n_prompt), -100, dtype=token_labels.dtype).to(device)
            labels = torch.cat([prompt_labels, token_labels], dim=1)

            # Shift for causal LM: compare logits[:, :-1] with labels[:, 1:]
            loss = loss_fn(
                logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                labels[:, 1:].contiguous().view(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_data):.4f}")

    return model

# Train လုပ်ခြင်း
trained_model = train_prompt_tuning(model, train_data)

# Soft prompt သိမ်းဆည်းခြင်း
torch.save(model.soft_prompt, "ros2_soft_prompt.pt")
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

### P-Tuning v1 Implementation

```python
class PTuning(nn.Module):
    def __init__(self, model_name, n_prompt_tokens=10, prompt_positions=None):
        super().__init__()
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Freeze model parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        embedding_dim = self.model.config.hidden_size
        
        # LSTM encoder for soft prompts (P-Tuning v1 ၏ အထူးအချက်)
        self.prompt_encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        
        # Prompt embeddings
        self.soft_prompts = nn.Parameter(
            torch.randn(n_prompt_tokens, embedding_dim)
        )
        
        # Projection layer
        self.projection = nn.Linear(embedding_dim * 2, embedding_dim)
        
        self.n_prompt_tokens = n_prompt_tokens
        self.prompt_positions = prompt_positions or list(range(n_prompt_tokens))
    
    def get_prompt_embeddings(self, batch_size):
        # LSTM ဖြင့် soft prompts များကို encode လုပ်ခြင်း
        prompts = self.soft_prompts.unsqueeze(0).expand(batch_size, -1, -1)
        encoded_prompts, _ = self.prompt_encoder(prompts)
        
        # Project back to embedding dimension
        prompt_embeddings = self.projection(encoded_prompts)
        
        return prompt_embeddings
    
    def forward(self, input_ids, attention_mask=None):
        batch_size = input_ids.shape[0]
        
        # Input embeddings
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        
        # Prompt embeddings ရယ်ခြင်း
        prompt_embeds = self.get_prompt_embeddings(batch_size)
        
        # Interleave prompts with input (simplified version)
        # Real implementation တွင် prompt_positions အတိုင်း ထည့်ရမည်
        inputs_embeds = torch.cat([prompt_embeds, inputs_embeds], dim=1)
        
        # Attention mask update
        if attention_mask is not None:
            prefix_attention = torch.ones(
                batch_size, self.n_prompt_tokens
            ).to(attention_mask.device)
            attention_mask = torch.cat([prefix_attention, attention_mask], dim=1)
        
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        )
        
        return outputs

# အသုံးပြုနည်း
p_tuning_model = PTuning("gpt2", n_prompt_tokens=10)

print(f"Trainable params: {sum(p.numel() for p in p_tuning_model.parameters() if p.requires_grad)}")
```

### P-Tuning v2

P-Tuning v2 သည် prefix tuning နှင့် ပိုမို ဆင်တူပြီး every layer တွင် soft prompts များ ထည့်သည်။

```python
class PTuningV2(nn.Module):
    def __init__(self, model_name, n_prompt_tokens=20):
        super().__init__()
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Freeze model
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Layer-specific prompts
        num_layers = self.model.config.num_hidden_layers
        hidden_size = self.model.config.hidden_size
        
        # Every layer အတွက် soft prompts
        self.prompt_embeddings = nn.ParameterList([
            nn.Parameter(torch.randn(n_prompt_tokens, hidden_size))
            for _ in range(num_layers)
        ])
        
        self.n_prompt_tokens = n_prompt_tokens
    
    def forward(self, input_ids, attention_mask=None):
        # Implementation သည် model architecture ပေါ်မူတည်၍ ကွဲပြားသည်
        # ဤနေရာတွင် conceptual example သာ ဖြစ်သည်
        pass
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

## ROS2 Navigation အတွက် Soft Fine-Tuning

### Use Case: Natural Language to ROS2 Commands

```python
import torch
from peft import get_peft_model, PromptTuningConfig, TaskType

# Base model load လုပ်ခြင်း
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prompt Tuning configuration
peft_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=20,  # Soft prompt length
    prompt_tuning_init="RANDOM",  # or "TEXT"
)

# PEFT model ဖန်တီးခြင်း
peft_model = get_peft_model(model, peft_config)

# Trainable parameters ကြည့်ခြင်း
peft_model.print_trainable_parameters()
# Output: trainable params: 15,360 || all params: 124,454,400 || trainable%: 0.01234

# ROS2 dataset
train_examples = [
    {
        "instruction": "Move the robot forward at 0.5 m/s",
        "output": "ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \"{linear: {x: 0.5, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}\""
    },
    {
        "instruction": "Rotate the robot 180 degrees",
        "output": "ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \"{linear: {x: 0.0}, angular: {z: 3.14}}\""
    },
    {
        "instruction": "Stop the robot immediately",
        "output": "ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \"{}\""
    },
    {
        "instruction": "Navigate to the charging station",
        "output": "ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \"{pose: {pose: {position: {x: 5.0, y: 3.0}}}}\""
    }
]

# Training loop
from torch.optim import AdamW

optimizer = AdamW(peft_model.parameters(), lr=1e-3)

for epoch in range(10):
    total_loss = 0
    
    for example in train_examples:
        # Format input
        text = f"Instruction: {example['instruction']}\nCommand: {example['output']}"
        inputs = tokenizer(text, return_tensors="pt", max_length=256, truncation=True)
        
        # Forward pass
        outputs = peft_model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_examples):.4f}")

# Model သိမ်းဆည်းခြင်း
peft_model.save_pretrained("./ros2_prompt_tuned_model")
```

---

## Advantages of Soft Fine-Tuning

### ✅ အားသာချက်များ

1. **Parameter Efficient**: Model parameters အနည်းငယ်သာ train လုပ်ရသည်
   ```
   Full Fine-tuning: 124M parameters
   Prompt Tuning: 15K parameters (99.99% reduction!)
   ```

2. **Memory Efficient**: GPU memory သက်သာစွာ အသုံးပြုနိုင်သည်

3. **Multi-Task**: Task တစ်ခုချင်းစီအတွက် soft prompts သီးခြား သိမ်းနိုင်သည်
   ```
   Base Model (shared): 124M params
   Task 1 soft prompt: 15K
   Task 2 soft prompt: 15K
   Task 3 soft prompt: 15K
   ```

4. **Faster Training**: Parameters နည်းသောကြောင့် မြန်ဆန်သည်

5. **No Catastrophic Forgetting**: Base model ကို မပြောင်းလဲသောကြောင့် original capabilities ဆုံးရှုံးမှု မရှိ

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
