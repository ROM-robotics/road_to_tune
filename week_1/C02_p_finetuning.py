import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# 1. CONFIG
# ============================================================
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

N_PROMPT_TOKENS = 20
LR = 5e-5
EPOCHS = 150
MAX_NEW_TOKENS = 64

# ============================================================
# 2. TRAINING DATA
# ============================================================
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

# ============================================================
# 3. LOAD MODEL & TOKENIZER
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=DTYPE,
    device_map="auto"
)

for p in model.parameters():
    p.requires_grad = False

model.eval()

# ============================================================
# 4. P-TUNING v2 PROMPT ENCODER (DTYPE SAFE)
# ============================================================
class PTuningV2Prompt(nn.Module):
    def __init__(self, n_tokens, hidden_size, dtype):
        super().__init__()

        self.virtual_tokens = torch.arange(n_tokens)

        self.embedding = nn.Embedding(n_tokens, hidden_size, dtype=dtype)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, dtype=dtype),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size, dtype=dtype)
        )

        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, batch_size, device):
        tokens = self.virtual_tokens.to(device)
        x = self.embedding(tokens)
        x = self.mlp(x)
        return x.unsqueeze(0).expand(batch_size, -1, -1)

prompt_encoder = PTuningV2Prompt(
    N_PROMPT_TOKENS,
    model.config.hidden_size,
    DTYPE
).to(model.device)

# ============================================================
# 5. LOSS FUNCTION
# ============================================================
def compute_loss(input_text, target_text):
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
    target_ids = tokenizer(target_text, return_tensors="pt").input_ids.to(model.device)

    batch_size = input_ids.size(0)

    full_ids = torch.cat([input_ids, target_ids], dim=1)

    token_embeds = model.get_input_embeddings()(full_ids).to(DTYPE)
    prompt_embeds = prompt_encoder(batch_size, model.device)

    full_embeds = torch.cat([prompt_embeds, token_embeds], dim=1)

    attention_mask = torch.ones(
        full_embeds.shape[:-1],
        device=model.device,
        dtype=torch.long
    )

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

    outputs = model(
        inputs_embeds=full_embeds,
        attention_mask=attention_mask,
        labels=labels
    )
    return outputs.loss

# ============================================================
# 6. TRAINING
# ============================================================
optimizer = torch.optim.AdamW(prompt_encoder.parameters(), lr=LR)

print("🚀 Starting P-Tuning v2 (dtype-safe)...\n")

for epoch in range(EPOCHS):
    total_loss = 0.0

    for inp, out in train_data:
        loss = compute_loss(inp, out)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prompt_encoder.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1:03d} | Loss: {total_loss:.4f}")

# ============================================================
# 7. SAVE PROMPT
# ============================================================
torch.save(prompt_encoder.state_dict(), "p_tuning_ros2.pt")
print("\n✅ Saved: p_tuning_ros2.pt")

# ============================================================
# 8. INFERENCE
# ============================================================
def infer_ros2_command(human_input):
    input_ids = tokenizer(human_input, return_tensors="pt").input_ids.to(model.device)
    input_embeds = model.get_input_embeddings()(input_ids).to(DTYPE)

    prompt_embeds = prompt_encoder(1, model.device)
    full_embeds = torch.cat([prompt_embeds, input_embeds], dim=1)

    with torch.no_grad():
        output_ids = model.generate(
            inputs_embeds=full_embeds,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# ============================================================
# 9. TEST
# ============================================================
print("\n🧪 Testing\n")

tests = [
    "Move forward 2 meters",
    "Turn left 90 degrees",
    "Navigate to waypoint A"
]

for t in tests:
    print("Input :", t)
    print("Output:", infer_ros2_command(t))
    print("-" * 80)
