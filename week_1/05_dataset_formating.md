# Dataset Formatting: Instruction + Output Format

## မိတ်ဆက်

LLM Fine-Tuning တွင် dataset formatting သည် အလွန်အရေးကြီးသည်။ Data ကို မှန်ကန်စွာ format လုပ်ခြင်းသည် model ၏ learning efficiency နှင့် final performance ကို သိသိသာသာ သက်ရောက်စေသည်။

## Dataset Format Types

### ၁. Instruction Format (အခြေခံဆုံး)

```json
{
  "instruction": "Move the robot forward at 0.5 m/s",
  "output": "ros2 topic pub /cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.5}}'"
}
```

### ၂. Instruction + Input + Output Format

```json
{
  "instruction": "Generate ROS2 command for the following robot action:",
  "input": "Move forward at 0.5 m/s",
  "output": "ros2 topic pub /cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.5}}'"
}
```

### ၃. Conversation Format (Multi-turn)

```json
{
  "conversations": [
    {"role": "user", "content": "How do I move my robot?"},
    {"role": "assistant", "content": "Use ros2 topic pub /cmd_vel..."},
    {"role": "user", "content": "What about turning?"},
    {"role": "assistant", "content": "For turning, modify angular.z..."}
  ]
}
```

---

## Single Text Format (Language Modeling အတွက်)

### Template Design

Model ကို train လုပ်ရာတွင် instruction နှင့် output ကို single text အဖြစ် ပေါင်းရသည်။

#### Basic Template
```python
template = """### Instruction:
{instruction}

### Response:
{output}"""
```

#### With Input Field
```python
template = """### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""
```

#### Alpaca Format
```python
alpaca_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}"""
```

---

## လက်တွေ့ Implementation

### ၁. ROS2 Command Generation Dataset

```python
import json
from typing import List, Dict

class ROS2DatasetFormatter:
    def __init__(self, template_type="basic"):
        self.templates = {
            "basic": """### Instruction:
{instruction}

### Command:
{output}""",
            
            "detailed": """### Task:
{instruction}

### Context:
{input}

### ROS2 Command:
{output}""",
            
            "conversational": """User: {instruction}
Assistant: {output}"""
        }
        
        self.template_type = template_type
    
    def format_single(self, instruction: str, output: str, input_text: str = "") -> str:
        """Single example ကို format လုပ်ခြင်း"""
        template = self.templates[self.template_type]
        
        if "{input}" in template and input_text:
            return template.format(
                instruction=instruction,
                input=input_text,
                output=output
            )
        else:
            return template.format(
                instruction=instruction,
                output=output
            )
    
    def format_dataset(self, data: List[Dict]) -> List[str]:
        """Dataset တစ်ခုလုံးကို format လုပ်ခြင်း"""
        formatted_data = []
        
        for item in data:
            formatted = self.format_single(
                instruction=item["instruction"],
                output=item["output"],
                input_text=item.get("input", "")
            )
            formatted_data.append(formatted)
        
        return formatted_data

# အသုံးပြုနည်း
raw_data = [
    {
        "instruction": "Move robot forward at 0.5 m/s",
        "output": "ros2 topic pub /cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.5}}'"
    },
    {
        "instruction": "Stop the robot immediately",
        "output": "ros2 topic pub /cmd_vel geometry_msgs/msg/Twist '{}'"
    },
    {
        "instruction": "Rotate robot counterclockwise",
        "output": "ros2 topic pub /cmd_vel geometry_msgs/msg/Twist '{angular: {z: 1.57}}'"
    }
]

formatter = ROS2DatasetFormatter(template_type="basic")
formatted_texts = formatter.format_dataset(raw_data)

for text in formatted_texts:
    print(text)
    print("-" * 50)
```

**Output:**
```
### Instruction:
Move robot forward at 0.5 m/s

### Command:
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.5}}'
--------------------------------------------------
### Instruction:
Stop the robot immediately

### Command:
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist '{}'
--------------------------------------------------
```

---

### ၂. Dataset Loading with HuggingFace Datasets

```python
from datasets import Dataset, DatasetDict
import pandas as pd

# Raw data ပြင်ဆင်ခြင်း
data = {
    "instruction": [
        "Move forward 2 meters",
        "Turn right 90 degrees",
        "Navigate to charging station",
        "Check battery status",
        "Stop all motors"
    ],
    "output": [
        "ros2 topic pub /cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.5}}' & sleep 4 && ros2 topic pub /cmd_vel geometry_msgs/msg/Twist '{}'",
        "ros2 topic pub /cmd_vel geometry_msgs/msg/Twist '{angular: {z: -1.57}}' & sleep 1 && ros2 topic pub /cmd_vel geometry_msgs/msg/Twist '{}'",
        "ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose '{pose: {pose: {position: {x: 5.0, y: 3.0}}}}'",
        "ros2 topic echo /battery_state sensor_msgs/msg/BatteryState",
        "ros2 topic pub /cmd_vel geometry_msgs/msg/Twist '{}'"
    ]
}

# Dataset ဖန်တီးခြင်း
df = pd.DataFrame(data)
dataset = Dataset.from_pandas(df)

# Train/validation split
dataset_dict = dataset.train_test_split(test_size=0.2, seed=42)

print(dataset_dict)
print("\nExample:")
print(dataset_dict["train"][0])
```

---

### ၃. Tokenization with Formatting

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def format_and_tokenize(examples, formatter):
    """Format ပြုလုပ်ပြီး tokenize လုပ်ခြင်း"""
    
    # Format လုပ်ခြင်း
    texts = []
    for inst, out in zip(examples["instruction"], examples["output"]):
        formatted = formatter.format_single(inst, out)
        texts.append(formatted)
    
    # Tokenize လုပ်ခြင်း
    tokenized = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )
    
    # Labels သတ်မှတ်ခြင်း
    tokenized["labels"] = tokenized["input_ids"].clone()
    
    return tokenized

# Apply to dataset
formatter = ROS2DatasetFormatter(template_type="basic")

tokenized_dataset = dataset_dict.map(
    lambda x: format_and_tokenize(x, formatter),
    batched=True,
    remove_columns=dataset_dict["train"].column_names
)

print(tokenized_dataset)
```

---

## Special Tokens & Masking

### Loss Calculation အတွက် Masking

Instruction part ကို loss calculation တွင် ignore လုပ်ရန် -100 သုံး၍ mask လုပ်နိုင်သည်။

```python
def format_with_masking(examples, tokenizer, formatter):
    """Instruction part ကို mask လုပ်ပြီး output part ကိုသာ loss တွက်ခြင်း"""
    
    formatted_texts = []
    
    for inst, out in zip(examples["instruction"], examples["output"]):
        # Instruction part
        instruction_text = f"### Instruction:\n{inst}\n\n### Command:\n"
        # Full text
        full_text = instruction_text + out
        
        formatted_texts.append({
            "full_text": full_text,
            "instruction_length": len(tokenizer.encode(instruction_text))
        })
    
    # Tokenize full texts
    full_texts = [item["full_text"] for item in formatted_texts]
    tokenized = tokenizer(
        full_texts,
        padding="max_length",
        truncation=True,
        max_length=256
    )
    
    # Labels ဖန်တီးပြီး instruction part ကို mask လုပ်ခြင်း
    labels = []
    for idx, item in enumerate(formatted_texts):
        label = tokenized["input_ids"][idx].copy()
        # Instruction part ကို -100 ထည့်ခြင်း (ignored in loss)
        inst_length = item["instruction_length"]
        label[:inst_length] = [-100] * inst_length
        labels.append(label)
    
    tokenized["labels"] = labels
    
    return tokenized

# Apply masking
masked_dataset = dataset_dict.map(
    lambda x: format_with_masking(x, tokenizer, formatter),
    batched=True,
    remove_columns=dataset_dict["train"].column_names
)
```

---

## Advanced Formatting Techniques

### ၁. Context Window Management

```python
def smart_truncation(text: str, tokenizer, max_length: int = 512):
    """Important parts ကို ထိန်းသိမ်းထားပြီး truncate လုပ်ခြင်း"""
    
    tokens = tokenizer.encode(text)
    
    if len(tokens) <= max_length:
        return text
    
    # Instruction နှင့် response start ကို ထိန်းသိမ်းခြင်း
    # Middle part ကို truncate လုပ်ခြင်း
    instruction_end = text.find("### Command:")
    
    if instruction_end == -1:
        # Simple truncation
        return tokenizer.decode(tokens[:max_length])
    
    instruction_part = text[:instruction_end + len("### Command:\n")]
    response_part = text[instruction_end + len("### Command:\n"):]
    
    inst_tokens = tokenizer.encode(instruction_part)
    remaining_length = max_length - len(inst_tokens) - 10  # Buffer
    
    response_tokens = tokenizer.encode(response_part)[:remaining_length]
    truncated_response = tokenizer.decode(response_tokens)
    
    return instruction_part + truncated_response
```

---

### ၂. Data Augmentation

```python
def augment_ros2_commands(data: List[Dict]) -> List[Dict]:
    """ROS2 commands များကို augment လုပ်ခြင်း"""
    
    augmented = []
    
    for item in data:
        # Original
        augmented.append(item)
        
        # Variation 1: Different speed values
        if "linear: {x:" in item["output"]:
            for speed in [0.3, 0.7, 1.0]:
                new_item = item.copy()
                new_item["instruction"] = item["instruction"].replace("0.5", str(speed))
                new_item["output"] = item["output"].replace("0.5", str(speed))
                augmented.append(new_item)
        
        # Variation 2: Rephrase instructions
        rephrasings = {
            "Move forward": ["Go forward", "Advance", "Drive ahead"],
            "Turn left": ["Rotate left", "Turn counterclockwise"],
            "Stop": ["Halt", "Freeze", "Emergency stop"]
        }
        
        for original, variations in rephrasings.items():
            if original in item["instruction"]:
                for variant in variations:
                    new_item = item.copy()
                    new_item["instruction"] = item["instruction"].replace(original, variant)
                    augmented.append(new_item)
    
    return augmented

# Augmentation အသုံးပြုခြင်း
augmented_data = augment_ros2_commands(raw_data)
print(f"Original: {len(raw_data)} → Augmented: {len(augmented_data)}")
```

---

### ၃. Multi-Turn Conversation Format

```python
def format_conversation(messages: List[Dict]) -> str:
    """Multi-turn conversation ကို format လုပ်ခြင်း"""
    
    formatted = ""
    
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        
        if role == "system":
            formatted += f"System: {content}\n\n"
        elif role == "user":
            formatted += f"User: {content}\n\n"
        elif role == "assistant":
            formatted += f"Assistant: {content}\n\n"
    
    return formatted.strip()

# Example conversation
conversation = [
    {
        "role": "system",
        "content": "You are a ROS2 expert assistant."
    },
    {
        "role": "user",
        "content": "How do I check active topics?"
    },
    {
        "role": "assistant",
        "content": "Use: ros2 topic list"
    },
    {
        "role": "user",
        "content": "How about nodes?"
    },
    {
        "role": "assistant",
        "content": "Use: ros2 node list"
    }
]

formatted_conv = format_conversation(conversation)
print(formatted_conv)
```

---

## JSON/JSONL Format

### JSONL (JSON Lines) for Large Datasets

```python
import json

def save_to_jsonl(data: List[Dict], filename: str):
    """JSONL format တွင် သိမ်းဆည်းခြင်း"""
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def load_from_jsonl(filename: str) -> List[Dict]:
    """JSONL file မှ ဖတ်ခြင်း"""
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

# သိမ်းဆည်းခြင်း
save_to_jsonl(raw_data, "ros2_commands.jsonl")

# ပြန်ဖတ်ခြင်း
loaded_data = load_from_jsonl("ros2_commands.jsonl")
print(f"Loaded {len(loaded_data)} examples")
```

---

## Quality Checks

### Dataset Validation

```python
def validate_dataset(data: List[Dict]) -> Dict:
    """Dataset quality စစ်ဆေးခြင်း"""
    
    issues = {
        "empty_instructions": [],
        "empty_outputs": [],
        "too_long": [],
        "duplicates": []
    }
    
    seen = set()
    
    for idx, item in enumerate(data):
        # Empty check
        if not item.get("instruction", "").strip():
            issues["empty_instructions"].append(idx)
        
        if not item.get("output", "").strip():
            issues["empty_outputs"].append(idx)
        
        # Length check
        combined = item.get("instruction", "") + item.get("output", "")
        if len(combined) > 2000:
            issues["too_long"].append(idx)
        
        # Duplicate check
        key = (item.get("instruction", ""), item.get("output", ""))
        if key in seen:
            issues["duplicates"].append(idx)
        seen.add(key)
    
    # Report
    total_issues = sum(len(v) for v in issues.values())
    print(f"Total examples: {len(data)}")
    print(f"Total issues: {total_issues}")
    
    for issue_type, indices in issues.items():
        if indices:
            print(f"  {issue_type}: {len(indices)} cases")
    
    return issues

# Validate
issues = validate_dataset(raw_data)
```

---

## HuggingFace Datasets Format

### Upload to Hub

```python
from datasets import Dataset, DatasetDict

# Dataset ဖန်တီးခြင်း
train_dataset = Dataset.from_dict({
    "instruction": [...],
    "output": [...]
})

val_dataset = Dataset.from_dict({
    "instruction": [...],
    "output": [...]
})

dataset_dict = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset
})

# Hub သို့ push လုပ်ခြင်း
dataset_dict.push_to_hub("your-username/ros2-commands")

# Load လုပ်ခြင်း
from datasets import load_dataset
dataset = load_dataset("your-username/ros2-commands")
```

---

## လေ့ကျင့်ခန်း

### Exercise 1: Custom Formatter
```python
# TODO: ROS2 error messages အတွက် custom formatter တစ်ခု ရေးပါ
# Format: Error message → Explanation + Solution

class ROS2ErrorFormatter:
    def __init__(self):
        # TODO: Implement
        pass
    
    def format_error(self, error_msg: str, explanation: str, solution: str) -> str:
        # TODO: Implement
        pass

# Test data
errors = [
    {
        "error": "Failed to connect to /cmd_vel",
        "explanation": "The topic does not exist or no node is publishing to it",
        "solution": "Check with 'ros2 topic list' and ensure the robot driver is running"
    }
]
```

### Exercise 2: Data Augmentation
```python
# TODO: Nav2 waypoint commands များအတွက် data augmentation function ရေးပါ
# Generate variations with different coordinates

def augment_waypoints(base_command: str, num_variations: int = 5) -> List[str]:
    # TODO: Implement
    pass
```

### Exercise 3: Quality Filter
```python
# TODO: Dataset quality filter တစ်ခု implement လုပ်ပါ
# Remove low-quality examples based on:
# - Length (too short or too long)
# - Repetition
# - Invalid ROS2 commands

def filter_dataset(data: List[Dict]) -> List[Dict]:
    # TODO: Implement
    pass
```

---

## Best Practices

### ✅ လုပ်သင့်သည်များ

1. **Consistent Formatting**: Template တစ်မျိုးတည်းကို သုံးပါ
2. **Clear Separators**: Instruction နှင့် output ကို ရှင်းလင်းစွာ ခွဲခြားပါ
3. **Proper Tokenization**: Special tokens များကို မှန်မှန် သုံးပါ
4. **Data Validation**: Training မလုပ်ခင် quality check လုပ်ပါ
5. **Version Control**: Dataset versions များကို track လုပ်ပါ

### ❌ ရှောင်သင့်သည်များ

1. **Mixed Formats**: Dataset တစ်ခုတည်းတွင် format များ ရောထွေးခြင်း
2. **No Validation**: Quality check မလုပ်ဘဲ train လုပ်ခြင်း
3. **Too Long Sequences**: Context window ကျော်လွန်ခြင်း
4. **Duplicates**: Duplicate data များ ထည့်ခြင်း
5. **Inconsistent Labels**: Output format များ မတသမှတသ ဖြစ်ခြင်း

---

## အနှစ်ချုပ်

- **Template Design**: Clear instruction/output separation
- **Single Text Format**: Language modeling အတွက် text တစ်ခုတည်းအဖြစ် ပေါင်းရသည်
- **Masking**: Instruction part ကို loss calculation တွင် ignore လုပ်နိုင်သည်
- **Quality Matters**: Good data → Good model
- **HuggingFace Integration**: Datasets library သုံး၍ efficient data handling

**Key Takeaway**: Dataset formatting သည် model performance ၏ 50% ကို သက်ရောက်စေနိုင်သည်။ Time ယူပြီး မှန်ကန်စွာ format လုပ်ပါ။

နောက်ဆုံးသင်ခန်းစာတွင် ROS2/Nav2 use-cases များကို prompt အဖြစ် ခွဲခြားသတ်မှတ်ခြင်းကို လေ့လာပါမည်။

