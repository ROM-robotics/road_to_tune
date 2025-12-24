# -------------------------------------------------------------------
# 01 ROS2 Command Dataset ပြင်ဆင်ခြင်း
# -------------------------------------------------------------------
from datasets import Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType


# -------------------------------------------------------------------
# 02 Dataset
# -------------------------------------------------------------------
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

dataset = Dataset.from_dict(data)
dataset = dataset.train_test_split(test_size=0.2)
print("Data { train, Test -structure }",dataset)


# -------------------------------------------------------------------
# 03 Tokenizer & Tokenization
# -------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct")

# Qwen model တချို့မှာ pad_token မရှိလို့,Padding လုပ်တဲ့အခါ သုံးမယ့် pad_token ကို eos_token နဲ့တူအောင် သတ်မှတ်ပေးတာပါ။
tokenizer.pad_token = tokenizer.eos_token  

# dimension တူအောင် padding လုပ်ဖို့ လိုအပ်တဲ့အဆင့်တစ်ခု ဖြစ်ပါတယ်။
print("tokenizer.pad_token :", tokenizer.pad_token)         


def tokenize_function(examples):
    texts = [
        f"### Instruction:\n{inst}\n\n### Command:\n{out}"
        for inst, out in zip(examples["instruction"], examples["output"])
    ]

    tokenized = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=256,
    )

    labels = []
    for ids in tokenized["input_ids"]:
        labels.append([
            token if token != tokenizer.pad_token_id else -100
            for token in ids
        ])

    tokenized["labels"] = labels

    print("\ninput_ids: { tokenize_function }", tokenized["input_ids"])
    print("\nlabels: { tokenize_function }", tokenized["labels"])
    print("\ninput_ids length: { tokenize_function }", [len(ids) for ids in tokenized["input_ids"]])
    print("\npad_token_id: { tokenize_function }", tokenizer.pad_token_id)

    return tokenized


tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)

print("tokenized_dataset { tokenize_function }: ",tokenized_dataset)


# -------------------------------------------------------------------
# 04 LoRA Configuration
# -------------------------------------------------------------------
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
)


# -------------------------------------------------------------------
# 05 Base Model → PEFT Model  
# -------------------------------------------------------------------
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    device_map="auto"
)

peft_model = get_peft_model(base_model, lora_config)
peft_model.print_trainable_parameters()


# -------------------------------------------------------------------
# 06 Training
# -------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir="./ros2_lora_model",
    num_train_epochs=10,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,

    logging_strategy="steps",
    logging_steps=1,
    eval_strategy="epoch",
    save_strategy="epoch",

    fp16=True,
    report_to="none",
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
)

print("Starting training...")
trainer.train()

trainer.save_model("./ros2_command_model_final")
print("Training complete!")
