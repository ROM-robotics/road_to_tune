# Week 2.2: Local Model Running နှင့် Prompt Tuning Advanced

## 1. Qwen / LLaMA Model ကို Local Environment မှာ Run လုပ်ခြင်း

### Qwen Model Local Run
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Model ကို local မှာ load လုပ်ခြင်း
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # GPU ရှိရင် auto detect
    torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Text generation
prompt = "Explain ROS2 navigation in simple terms:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### LLaMA Model Local Run (CPU/GPU)
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "meta-llama/Llama-3.2-1B"

# CPU အတွက်
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="cpu"
)

# GPU အတွက် (ရှိရင်)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.float16,
#     device_map="cuda:0"
# )

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
```

---

## 2. Virtual Tokens (20-50) နဲ့ Prompt Tuning Training

### Prompt Tuning Configuration
```python
from peft import PromptTuningConfig, get_peft_model, TaskType

# 30 virtual tokens သုံးပြီး prompt tuning setup
config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=30,  # 20-50 အကြား
    prompt_tuning_init="TEXT",  # or "RANDOM"
    prompt_tuning_init_text="Generate ROS2 navigation parameters:",
    tokenizer_name_or_path=model_name
)

# PEFT model ပြုလုပ်ခြင်း
peft_model = get_peft_model(model, config)
print(f"Trainable parameters: {peft_model.print_trainable_parameters()}")
```

### Training Example
```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./prompt_tuning_output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=3e-2,  # Prompt tuning က higher LR သုံးနိုင်တယ်
    logging_steps=10,
    save_steps=100
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

trainer.train()
```

---

## 3. CPU / GPU Environment အတွက် Training Config ချိန်ညှိခြင်း

### CPU Configuration
```python
from transformers import TrainingArguments

cpu_training_args = TrainingArguments(
    output_dir="./results_cpu",
    num_train_epochs=2,
    per_device_train_batch_size=1,  # သေးသေးသုံးပါ
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,  # Batch size ကို simulate လုပ်ဖို့
    learning_rate=5e-5,
    fp16=False,  # CPU က fp16 မသုံးနိုင်
    logging_steps=50,
    save_strategy="epoch",
    dataloader_num_workers=2,  # CPU cores အလိုက်
    no_cuda=True  # GPU မသုံးဘူးလို့ specify
)
```

### GPU Configuration
```python
gpu_training_args = TrainingArguments(
    output_dir="./results_gpu",
    num_train_epochs=3,
    per_device_train_batch_size=8,  # GPU ရှိတော့ ပိုများနိုင်
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=3e-5,
    fp16=True,  # GPU speed တက်ဖို့
    logging_steps=20,
    save_strategy="steps",
    save_steps=200,
    eval_strategy="steps",
    eval_steps=200,
    warmup_steps=100
)
```

### Auto Device Selection
```python
import torch

# Device ကို auto detect လုပ်ခြင်း
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Training args ကို dynamic ပြုလုပ်ခြင်း
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8 if device == "cuda" else 2,
    fp16=True if device == "cuda" else False,
    no_cuda=False if device == "cuda" else True
)
```

---

## 4. Output Format Consistency (YAML, XML, Launch Files) ကို Evaluate လုပ်ခြင်း

### YAML Format Validation
```python
import yaml
import re

def evaluate_yaml_format(generated_text):
    """YAML format မှန်ကန်မှုကို စစ်ဆေးခြင်း"""
    try:
        # YAML block ကို extract လုပ်ခြင်း
        yaml_match = re.search(r'```yaml\n(.*?)\n```', generated_text, re.DOTALL)
        if yaml_match:
            yaml_content = yaml_match.group(1)
            parsed = yaml.safe_load(yaml_content)
            return True, "Valid YAML", parsed
        return False, "No YAML block found", None
    except yaml.YAMLError as e:
        return False, f"Invalid YAML: {str(e)}", None

# Example usage
generated = """
Here is the configuration:
```yaml
waypoints:
  - x: 0.0
    y: 0.0
    z: 0.0
  - x: 1.0
    y: 1.0
    z: 0.0
```
"""

is_valid, message, parsed = evaluate_yaml_format(generated)
print(f"Valid: {is_valid}, Message: {message}")
```

### XML Format Validation
```python
import xml.etree.ElementTree as ET

def evaluate_xml_format(generated_text):
    """XML format မှန်ကန်မှုကို စစ်ဆေးခြင်း"""
    try:
        xml_match = re.search(r'```xml\n(.*?)\n```', generated_text, re.DOTALL)
        if xml_match:
            xml_content = xml_match.group(1)
            root = ET.fromstring(xml_content)
            return True, "Valid XML", root
        return False, "No XML block found", None
    except ET.ParseError as e:
        return False, f"Invalid XML: {str(e)}", None
```

### ROS2 Launch File Validation
```python
def evaluate_launch_format(generated_text):
    """ROS2 launch file format စစ်ဆေးခြင်း"""
    required_imports = [
        "from launch import LaunchDescription",
        "from launch_ros.actions import Node"
    ]
    
    has_launch_description = "LaunchDescription" in generated_text
    has_generate_launch = "def generate_launch_description" in generated_text
    has_return = "return LaunchDescription" in generated_text
    
    score = sum([
        has_launch_description,
        has_generate_launch,
        has_return
    ]) / 3.0
    
    return score, {
        "has_launch_description": has_launch_description,
        "has_generate_launch": has_generate_launch,
        "has_return": has_return
    }

# Example
launch_code = """
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(package='nav2_bt_navigator', executable='bt_navigator')
    ])
"""
score, details = evaluate_launch_format(launch_code)
print(f"Format Score: {score:.2%}, Details: {details}")
```

### Combined Evaluation Metrics
```python
def evaluate_output_consistency(model, tokenizer, test_prompts):
    """Multiple format outputs ကို consistent စစ်ဆေးခြင်း"""
    results = {
        "yaml_valid": 0,
        "xml_valid": 0,
        "launch_valid": 0,
        "total": len(test_prompts)
    }
    
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_length=512)
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Format အလိုက် validate လုပ်ခြင်း
        if "yaml" in prompt.lower():
            is_valid, _, _ = evaluate_yaml_format(generated)
            if is_valid:
                results["yaml_valid"] += 1
        elif "xml" in prompt.lower():
            is_valid, _, _ = evaluate_xml_format(generated)
            if is_valid:
                results["xml_valid"] += 1
        elif "launch" in prompt.lower():
            score, _ = evaluate_launch_format(generated)
            if score > 0.7:
                results["launch_valid"] += 1
    
    return results
```

---

## 5. ROS Nav2 Waypoint YAML, Parameter Explanation Dataset သင်ခြင်း

### Dataset Preparation
```python
from datasets import Dataset

# ROS Nav2 specific training data
nav2_data = [
    {
        "instruction": "Generate waypoint YAML for navigation",
        "output": """```yaml
waypoints:
  - header:
      frame_id: "map"
    pose:
      position:
        x: 2.0
        y: 3.0
        z: 0.0
      orientation:
        w: 1.0
```"""
    },
    {
        "instruction": "Explain planner_server parameters",
        "output": """The planner_server parameters:
- expected_planner_frequency: Planning rate (Hz)
- planner_plugins: List of planning algorithms
- GridBased.plugin: Type of planner (e.g., NavFn, SmacPlanner)
- GridBased.tolerance: Goal tolerance in meters"""
    },
    {
        "instruction": "Create launch file for Nav2",
        "output": """```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='nav2_planner',
            executable='planner_server',
            name='planner_server',
            parameters=[{'use_sim_time': True}]
        )
    ])
```"""
    }
]

# Dataset ပြုလုပ်ခြင်း
dataset = Dataset.from_list(nav2_data)
```

### Training with Format Instructions
```python
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

# Format instruction template
def format_instruction(example):
    return f"<|im_start|>user\n{example['instruction']}<|im_end|>\n<|im_start|>assistant\n{example['output']}<|im_end|>"

# Dataset ကို tokenize လုပ်ခြင်း
def tokenize_function(examples):
    formatted = [format_instruction(ex) for ex in examples]
    return tokenizer(formatted, truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Training
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_dataset
)

trainer.train()
```

### Inference နှင့် Evaluation
```python
def generate_nav2_output(prompt, model, tokenizer):
    """Nav2 specific output generation"""
    formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_length=512,
        temperature=0.7,
        do_sample=True,
        top_p=0.9
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test
test_prompts = [
    "Generate waypoint YAML for 3 positions",
    "Explain controller_server parameters",
    "Create launch file for navigation stack"
]

for prompt in test_prompts:
    result = generate_nav2_output(prompt, peft_model, tokenizer)
    print(f"\nPrompt: {prompt}")
    print(f"Output: {result}")
    print("-" * 50)
```

### Evaluation Metrics
```python
def evaluate_nav2_outputs(model, tokenizer, test_dataset):
    """Nav2 specific evaluation"""
    metrics = {
        "yaml_format_accuracy": 0,
        "explanation_relevance": 0,
        "launch_file_validity": 0
    }
    
    for item in test_dataset:
        generated = generate_nav2_output(item["instruction"], model, tokenizer)
        
        # YAML validity
        if "waypoint" in item["instruction"].lower():
            is_valid, _, _ = evaluate_yaml_format(generated)
            metrics["yaml_format_accuracy"] += int(is_valid)
        
        # Launch file validity
        elif "launch" in item["instruction"].lower():
            score, _ = evaluate_launch_format(generated)
            metrics["launch_file_validity"] += score
        
        # Explanation relevance (simple keyword matching)
        elif "explain" in item["instruction"].lower():
            keywords = ["parameter", "configuration", "setting"]
            relevance = sum(kw in generated.lower() for kw in keywords) / len(keywords)
            metrics["explanation_relevance"] += relevance
    
    # Average ထုတ်ခြင်း
    n = len(test_dataset)
    return {k: v/n for k, v in metrics.items()}

# Run evaluation
results = evaluate_nav2_outputs(peft_model, tokenizer, test_dataset)
print("Evaluation Results:", results)
```

---

## Summary

### Key Points:
1. **Local Model Running**: CPU/GPU အတွက် device_map နှင့် torch_dtype သုံးပြီး optimize လုပ်နိုင်
2. **Prompt Tuning**: 20-50 virtual tokens သုံးပြီး parameter efficient training
3. **Training Config**: CPU/GPU အလိုက် batch size, fp16, gradient accumulation ချိန်ညှိရမည်
4. **Format Consistency**: YAML, XML, Launch file တို့ကို programmatically validate လုပ်ရမည်
5. **ROS Nav2 Training**: Domain-specific dataset ပြုလုပ်ပြီး format instruction based training လုပ်ရမည်

### Tips:
- CPU မှာ train လုပ်ရင် gradient_accumulation_steps တိုးပြီး batch size သေးသေးသုံးပါ
- Prompt tuning က full fine-tuning ထက် learning rate မြင့်နိုင်တယ် (3e-2)
- Output format စစ်ဖို့ validation functions တွေ ရေးထားပါ
- ROS2 specific training မှာ proper instruction format သုံးပါ
