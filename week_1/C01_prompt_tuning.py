import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# ===============================
# 1. CONFIG
# ===============================
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_PROMPT_TOKENS = 20
LR = 1e-4   
EPOCHS = 50
MAX_NEW_TOKENS = 64

# ===============================
# 2. TRAINING DATA
# ===============================
train_data = [
    (
        "Move forward 2 meters",
        "ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \"{linear: {x: 2.0}}\""
    ),
    (
        "Turn left 90 degrees",
        "ros2 service call /rotate_robot robot_msgs/srv/Rotate \"{angle: 1.57}\""
    ),
    (
        "Navigate to waypoint A",
        "ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose "
        "\"{pose: {header: {frame_id: 'map'}, pose: {position: {x: 5.0, y: 2.0}}}}\""
    )
]

# ===============================
# 3. LOAD MODEL & TOKENIZER
# ===============================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    device_map="auto"
)

for p in model.parameters():
    p.requires_grad = False

# Inference/စမ်းသပ်ချင်ရင် 
model.eval() 

# ===============================
# 4. SOFT PROMPT MODULE
# ===============================
class SoftPrompt(nn.Module):
    def __init__(self, n_tokens, embedding_layer):
        super().__init__()

        # vector table မှ row 20 ကို ယူပြီး initialize လုပ်ခြင်း . ([20, 1536])
        init_prompt = embedding_layer.weight[:n_tokens].detach().clone()

        # ([20, 1536])
        self.prompt_embeddings = nn.Parameter(init_prompt) 
        print("shape after initialization ( SoftPrompt::init ):", self.prompt_embeddings.shape)

    def forward(self, batch_size):
        blabla = self.prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        print("shape after unsqueeze ( SoftPrompt::forward ):", blabla.shape)
        return blabla

# vector table ရယူခြင်း row 151936, col 1536
embedding_layer = model.get_input_embeddings()

# torch.Size([151936, 1536])
print("shape after embedding_layer ( main ):",embedding_layer.weight.shape)

soft_prompt = SoftPrompt(
    N_PROMPT_TOKENS,
    embedding_layer
).to(embedding_layer.weight.device)


# ===============================
# 5. TRAINING FUNCTION (FIXED)
# ===============================
def compute_loss(input_text, target_text):
    # ([1, 6]) 6 = token count of "Move forward 2 meters"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
    target_ids = tokenizer(target_text, return_tensors="pt").input_ids.to(model.device)

    batch_size = input_ids.size(0)
    print("input id shape ( compute_loss ): ",input_ids.shape)


    # Concatenate tokens
    full_ids = torch.cat([input_ids, target_ids], dim=1)
    # input tokens = 6 , target tokens = 15 , full tokens = 21 , [1, 21]
    print("full id shape ( compute_loss ): ",full_ids.shape)

    # Embeddings ,  shape: [1, 21, 1536]
    token_embeds = model.get_input_embeddings()(full_ids)
    print("token embed shape ( compute_loss ): ",token_embeds.shape)

    # Soft Prompt Embeddings , shape: [1, 20, 1536]
    prompt_embeds = soft_prompt(batch_size)
    print("prompt embed shape ( compute_loss ): ",prompt_embeds.shape)

    # Final embeddings: [PROMPT][INPUT][TARGET]
    full_embeds = torch.cat([prompt_embeds, token_embeds], dim=1)
    print("full embed shape ( compute_loss ): ",full_embeds.shape)

    # 🔒 ATTENTION MASK (THIS FIXES NaN)
    # attention_mask = 1 ဆိုတာ "ဒီ token ကို mask မလုပ်ဘူး, သတိထားပါ" လို့ ဆိုလိုပါတယ်။
    attention_mask = torch.ones(
        full_embeds.size()[:-1],
        device=model.device,
        dtype=torch.long
    )
    print("attention mask shape ( compute_loss ): ",attention_mask.shape)

    # Labels: ignore prompt + input ,
    # shape: [1, 20 + 6 + 15] = [1, 41]
    # Model ကို training လုပ်တဲ့အခါ prompt နဲ့ input ကို loss မတွက်ဘူး၊ target (output) ကိုပဲ loss တွက်မယ်။
    labels = torch.cat(
        [
            torch.full(
                (batch_size, N_PROMPT_TOKENS + input_ids.size(1)),
                -100,
                device=model.device,
                dtype=torch.long
            ),
            target_ids
        ],
        dim=1
    )
    print("labels shape  ( compute_loss ) : ",labels.shape)

    outputs = model(
        inputs_embeds=full_embeds,
        attention_mask=attention_mask,
        labels=labels
    )
    return outputs.loss



# ===============================
# 6. TRAIN PROMPT (WITH BETTER OUTPUT)
# ===============================
optimizer = torch.optim.AdamW(soft_prompt.parameters(), lr=LR)

print("\n" + "="*40)
print("🚀 Starting Prompt Tuning Training Session")
print("="*40 + "\n")

for epoch in range(EPOCHS):
    total_loss = 0.0

    for inp, out in train_data:
        loss = compute_loss(inp, out)

        optimizer.zero_grad()
        loss.backward()

        # 🔒 Prevent FP16 explosions
        torch.nn.utils.clip_grad_norm_(soft_prompt.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1:02d} | Loss: {total_loss:.4f}\n")

print("✨ Training Complete!\n")


# ===============================
# 7. SAVE SOFT PROMPT
# ===============================
torch.save(soft_prompt.state_dict(), "soft_prompt_ros2.pt")
print("\n✅ Soft prompt saved as soft_prompt_ros2.pt")

# ===============================
# 8. INFERENCE
# ===============================
def infer_ros2_command(human_input):
    input_ids = tokenizer(human_input, return_tensors="pt").input_ids.to(model.device)

    input_embeds = model.get_input_embeddings()(input_ids)
    # input_embeds shape: (1, input_seq_len, embedding_dim)

    # (1) ဆိုတာ batch size ၁ ခုအတွက် soft prompt embeddings တောင်းတာပါ။
    # instance(batch_size) လို့ ခေါ်လိုက်တာနဲ့ forward(self, batch_size) ကို automatic ခေါ်ပေးပါတယ်။
    # feedforward
    prompt_embeds = soft_prompt(1)
    print("\n,prompt_embeds shape ( infer_ros2_command ):", prompt_embeds.shape)

    full_embeds = torch.cat([prompt_embeds, input_embeds], dim=1)
    print("full_embeds shape ( infer_ros2_command ):", full_embeds.shape)


    with torch.no_grad():
        output_ids = model.generate(
            inputs_embeds=full_embeds,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# ===============================
# 9. TEST
# ===============================
print("\n🧪 Testing Prompt-Tuned Model\n")

tests = [
    "Move forward 9 meters",
    "Turn left 90 degrees",
    "Navigate to waypoint A"
]

for t in tests:
    print("Input :", t)
    print("\nOutput:", infer_ros2_command(t))
    print("-" * 80)