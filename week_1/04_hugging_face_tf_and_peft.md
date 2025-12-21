# HuggingFace Transformers & PEFT Library

## မိတ်ဆက်

HuggingFace သည် modern NLP/LLM development အတွက် အဓိက ecosystem ဖြစ်သည်။ ဤသင်ခန်းစာတွင် Transformers library နှင့် PEFT (Parameter-Efficient Fine-Tuning) library များကို လက်တွေ့ကျကျ လေ့လာမည်ဖြစ်ပါသည်။

## HuggingFace Transformers Library

### Installation

```bash
# Basic installation
pip install transformers

# With PyTorch
pip install transformers torch

# With TensorFlow
pip install transformers tensorflow

# Complete installation
pip install transformers[torch] datasets accelerate evaluate
```

---

## ၁. Model Loading & Basic Usage

### Pretrained Models အသုံးပြုခြင်း

```python
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

# ၁.၁ Tokenizer Load လုပ်ခြင်း
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Text tokenize လုပ်ခြင်း
text = "The robot moves forward"
tokens = tokenizer(text, return_tensors="pt")

print("Input IDs:", tokens.input_ids)
print("Tokens:", tokenizer.convert_ids_to_tokens(tokens.input_ids[0]))

# ၁.၂ Model Load လုပ်ခြင်း
model = AutoModel.from_pretrained("gpt2")

# Inference
outputs = model(**tokens)
print("Output shape:", outputs.last_hidden_state.shape)

# ၁.၃ Generation Model
gen_model = AutoModelForCausalLM.from_pretrained("gpt2")

# Text generation
output_ids = gen_model.generate(
    tokens.input_ids,
    max_length=50,
    num_return_sequences=1,
    temperature=0.7,
    do_sample=True
)

generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("Generated:", generated_text)
```

---

### Model Types များ

```python
# Text Classification
from transformers import AutoModelForSequenceClassification

classifier = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=3
)

# Question Answering
from transformers import AutoModelForQuestionAnswering

qa_model = AutoModelForQuestionAnswering.from_pretrained(
    "distilbert-base-uncased-distilled-squad"
)

# Text Generation
from transformers import AutoModelForCausalLM

gpt_model = AutoModelForCausalLM.from_pretrained("gpt2")

# Sequence-to-Sequence
from transformers import AutoModelForSeq2SeqLM

t5_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
```

---

## ၂. Pipeline API (အလွယ်ဆုံး နည်းလမ်း)

### Text Generation Pipeline

```python
from transformers import pipeline

# ROS2 command generator pipeline
generator = pipeline("text-generation", model="gpt2")

prompt = "To move the robot forward in ROS2, use the command:"
result = generator(
    prompt,
    max_length=100,
    num_return_sequences=1,
    temperature=0.7
)

print(result[0]["generated_text"])
```

### Classification Pipeline

```python
# Sentiment analysis for robot feedback
classifier = pipeline("sentiment-analysis")

feedback_texts = [
    "The robot navigation is working perfectly!",
    "Robot keeps crashing into obstacles",
    "Average performance, needs improvement"
]

for text in feedback_texts:
    result = classifier(text)
    print(f"{text}\n→ {result}\n")
```

### Question Answering Pipeline

```python
# ROS2 documentation QA
qa_pipeline = pipeline("question-answering")

context = """
ROS2 uses the DDS (Data Distribution Service) middleware for communication.
Nodes can publish and subscribe to topics, call services, and use actions.
The command 'ros2 topic list' shows all active topics.
"""

question = "What command shows all active topics?"

answer = qa_pipeline(question=question, context=context)
print(f"Q: {question}")
print(f"A: {answer['answer']} (confidence: {answer['score']:.2f})")
```

---

## ၃. Trainer API for Fine-Tuning

### ROS2 Command Dataset ပြင်ဆင်ခြင်း

```python
from datasets import Dataset
import pandas as pd

# ROS2 command generation dataset
data = {
    "instruction": [
        "Move robot forward at 0.5 m/s",
        "Turn robot left 90 degrees",
        "Stop the robot",
        "Navigate to position x=2, y=3",
        "Rotate robot clockwise"
    ],
    "output": [
        "ros2 topic pub /cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.5}}'",
        "ros2 topic pub /cmd_vel geometry_msgs/msg/Twist '{angular: {z: 1.57}}'",
        "ros2 topic pub /cmd_vel geometry_msgs/msg/Twist '{}'",
        "ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose '{pose: {pose: {position: {x: 2.0, y: 3.0}}}}'",
        "ros2 topic pub /cmd_vel geometry_msgs/msg/Twist '{angular: {z: -1.57}}'"
    ]
}

# Dataset ဖန်တီးခြင်း
dataset = Dataset.from_dict(data)

# Train/Test split
dataset = dataset.train_test_split(test_size=0.2)
print(dataset)
```

### Tokenization Function

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    # Instruction + Output ကို ပေါင်းခြင်း
    texts = [
        f"### Instruction:\n{inst}\n\n### Command:\n{out}"
        for inst, out in zip(examples["instruction"], examples["output"])
    ]
    
    # Tokenize
    tokenized = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )
    
    # Labels သတ်မှတ်ခြင်း (for language modeling)
    tokenized["labels"] = tokenized["input_ids"].clone()
    
    return tokenized

# Apply tokenization
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)

print(tokenized_dataset)
```

### Training Configuration

```python
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

# Model load လုပ်ခြင်း
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Training arguments
training_args = TrainingArguments(
    output_dir="./ros2_command_model",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Causal LM အတွက်
)

# Trainer initialization
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
)

# Training စတင်ခြင်း
print("Starting training...")
trainer.train()

# Model သိမ်းဆည်းခြင်း
trainer.save_model("./ros2_command_model_final")
print("Training complete!")
```

---

## ၄. PEFT Library (Parameter-Efficient Fine-Tuning)

### Installation

```bash
pip install peft
```

### LoRA (Low-Rank Adaptation)

**သဘောတရား:** Model weights W ကို update မလုပ်ဘဲ low-rank matrices A နှင့် B ထည့်၍ ΔW = BA ကို train လုပ်သည်။

```python
from peft import LoraConfig, get_peft_model, TaskType

# Base model
base_model = AutoModelForCausalLM.from_pretrained("gpt2")

# LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,  # Rank
    lora_alpha=32,  # Scaling factor
    lora_dropout=0.1,
    target_modules=["c_attn", "c_proj"],  # GPT-2 attention layers
)

# PEFT model ဖန်တီးခြင်း
peft_model = get_peft_model(base_model, lora_config)

# Trainable parameters ကြည့်ခြင်း
peft_model.print_trainable_parameters()
# Output: trainable params: 294,912 || all params: 124,439,808 || trainable%: 0.2369
```

### LoRA Training

```python
from transformers import Trainer, TrainingArguments

# Training arguments
training_args = TrainingArguments(
    output_dir="./ros2_lora_model",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    learning_rate=1e-4,
    logging_steps=10,
    save_strategy="epoch",
)

# Trainer with LoRA model
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
)

# Training
trainer.train()

# LoRA weights သိမ်းဆည်းခြင်း
peft_model.save_pretrained("./ros2_lora_weights")
```

### Loading LoRA Model

```python
from peft import PeftModel

# Base model load လုပ်ခြင်း
base_model = AutoModelForCausalLM.from_pretrained("gpt2")

# LoRA weights merge လုပ်ခြင်း
peft_model = PeftModel.from_pretrained(
    base_model,
    "./ros2_lora_weights"
)

# Inference
tokenizer = AutoTokenizer.from_pretrained("gpt2")
prompt = "### Instruction:\nMove robot backward\n\n### Command:\n"
inputs = tokenizer(prompt, return_tensors="pt")

outputs = peft_model.generate(
    **inputs,
    max_length=200,
    temperature=0.7,
    do_sample=True
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

### Prompt Tuning with PEFT

```python
from peft import PromptTuningConfig, PromptTuningInit

# Prompt Tuning configuration
pt_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=20,  # Soft prompt length
    prompt_tuning_init=PromptTuningInit.RANDOM,
    tokenizer_name_or_path="gpt2",
)

# PEFT model
peft_model = get_peft_model(base_model, pt_config)
peft_model.print_trainable_parameters()
# Output: trainable params: 15,360 || all params: 124,454,400 || trainable%: 0.0123
```

---

### Prefix Tuning

```python
from peft import PrefixTuningConfig

# Prefix Tuning configuration
prefix_config = PrefixTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=30,
    encoder_hidden_size=768,  # GPT-2 hidden size
)

peft_model = get_peft_model(base_model, prefix_config)
peft_model.print_trainable_parameters()
```

---

## ၅. PEFT Methods Comparison

### Comparison Table

| Method | Parameters | Memory | Training Speed | Performance |
|--------|-----------|--------|----------------|-------------|
| Full Fine-Tuning | 100% | High | Slow | Best |
| LoRA | ~0.1-1% | Low | Fast | Very Good |
| Prefix Tuning | ~0.1% | Very Low | Very Fast | Good |
| Prompt Tuning | ~0.01% | Very Low | Very Fast | Good |
| Adapter Layers | ~1-5% | Medium | Medium | Very Good |

### ROS2 Application အတွက် ရွေးချယ်ပုံ

```python
# Scenario 1: Limited GPU memory (< 8GB)
# → Use Prompt Tuning or Prefix Tuning
config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=20
)

# Scenario 2: Need best performance (GPU >= 16GB)
# → Use LoRA
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
)

# Scenario 3: Multiple ROS2 tasks
# → Use separate adapters for each task
config_nav = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8)
config_debug = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8)
config_explain = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8)
```

---

## ၆. Advanced Features

### Gradient Checkpointing (Memory Optimization)

```python
model = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    gradient_checkpointing=True  # Memory သက်သာစေသည်
)
```

### Mixed Precision Training

```python
training_args = TrainingArguments(
    output_dir="./output",
    fp16=True,  # 16-bit floating point
    # or
    bf16=True,  # Brain float 16 (newer GPUs)
)
```

### Model Quantization

```python
from transformers import BitsAndBytesConfig

# 8-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True
)

model = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    quantization_config=quantization_config,
    device_map="auto"
)

# 4-bit quantization (အများဆုံး memory သက်သာသည်)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
```

---

## ၇. Saving & Loading Models

### HuggingFace Hub သို့ Upload လုပ်ခြင်း

```python
# Login to HuggingFace
from huggingface_hub import login
login(token="your_token_here")

# Model push လုပ်ခြင်း
model.push_to_hub("your-username/ros2-command-generator")
tokenizer.push_to_hub("your-username/ros2-command-generator")

# PEFT model push လုပ်ခြင်း
peft_model.push_to_hub("your-username/ros2-lora-adapter")
```

### Local သိမ်းဆည်းခြင်း

```python
# Full model
model.save_pretrained("./local_model")
tokenizer.save_pretrained("./local_model")

# PEFT adapter only
peft_model.save_pretrained("./local_adapter")
```

### Loading from Hub or Local

```python
# From Hub
model = AutoModelForCausalLM.from_pretrained("your-username/ros2-command-generator")

# From Local
model = AutoModelForCausalLM.from_pretrained("./local_model")

# PEFT adapter
from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained("gpt2")
peft_model = PeftModel.from_pretrained(base_model, "./local_adapter")
```

---

## လေ့ကျင့်ခန်း

### Exercise 1: Pipeline Exploration
```python
# TODO: ROS2 error messages များအတွက် sentiment analysis pipeline တစ်ခု ဖန်တီးပါ
# Test with different error messages and classify them

error_messages = [
    "Failed to connect to /cmd_vel topic",
    "Navigation successful, goal reached",
    "Warning: Robot approaching obstacle"
]

# TODO: Implement classifier
```

### Exercise 2: LoRA Fine-Tuning
```python
# TODO: ROS2 Q&A dataset ဖြင့် GPT-2 ကို LoRA သုံး၍ fine-tune လုပ်ပါ
# Requirements:
# - r=8, lora_alpha=32
# - Train for 3 epochs
# - Compare with base model

# Dataset example:
qa_data = [
    {"q": "What is a ROS2 node?", "a": "A node is a process that performs computation..."},
    # Add more...
]
```

### Exercise 3: Multiple Adapters
```python
# TODO: Tasks သုံးခု အတွက် LoRA adapters သုံးခု ဖန်တီးပါ
# Task 1: Command generation
# Task 2: Error explanation  
# Task 3: Code documentation

# ပြီးမှ adapter တစ်ခုချင်းစီကို load လုပ်ပြီး test လုပ်ပါ
```

---

## အနှစ်ချုပ်

- **Transformers Library**: Model loading, tokenization, training အတွက် complete ecosystem
- **Pipeline API**: အမြန်ဆုံး inference နည်းလမ်း
- **Trainer API**: Full fine-tuning အတွက် high-level interface
- **PEFT Library**: Memory-efficient fine-tuning methods
  - **LoRA**: Best balance (performance vs efficiency)
  - **Prompt Tuning**: Most parameter-efficient
  - **Prefix Tuning**: Good for generation tasks

**ROS2 Development အတွက် Recommendation:**
- Small models (GPT-2, DistilBERT) + LoRA = Real-time capable
- PEFT methods သုံး၍ multiple ROS2 tasks အတွက် adapters များ ဖန်တီးပါ
- HuggingFace Hub တွင် models များ share လုပ်ပြီး community နှင့် collaborate လုပ်ပါ

နောက်သင်ခန်းစာတွင် Dataset Formatting techniques များကို လေ့လာပါမည်။
