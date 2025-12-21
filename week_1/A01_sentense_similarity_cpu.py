import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel

# CPU only
device = torch.device("cpu")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.to(device)
model.eval()

sentences = [
    "The robot is moving forward",
    "The robot is stationary",
    "The autonomous vehicle navigates"
]

# Tokenize
inputs = tokenizer(
    sentences,
    padding=True,
    truncation=True,
    return_tensors="pt"
)

input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)

# Forward pass (no grad)
with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

# Mean pooling (attention-masked)
mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()  # [batch, seq_len, hidden]
masked_embeddings = last_hidden_state * mask

sum_embeddings = masked_embeddings.sum(dim=1)  # [batch, hidden]

# Use attention_mask to get token counts per example (shape [batch, 1])
token_counts = attention_mask.sum(dim=1).unsqueeze(-1).float()  # [batch, 1]
token_counts = token_counts.clamp(min=1e-9)  # avoid div by zero

sentence_embeddings = sum_embeddings / token_counts  # broadcasting: [batch, hidden] / [batch, 1]

# Normalize embeddings (recommended for cosine similarity)
sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

# Cosine similarity matrix (dot product since vectors are normalized)
similarity_matrix = torch.mm(sentence_embeddings, sentence_embeddings.t())

# Print readable results
sim_np = similarity_matrix.cpu().numpy()
labels = sentences

print("Pairwise cosine similarity matrix (rows/cols = sentences):\n")

# Header (trim long labels)
print(" " * 4, end="")
for lbl in labels:
    print(f"{lbl[:20]:>25}", end="")
print()

for i, row in enumerate(sim_np):
    print(f"{labels[i][:20]:<20}", end="")
    for v in row:
        print(f"{v:25.4f}", end="")
    print()

print("\nPairwise similarities:")
n = len(sentences)
for i in range(n):
    for j in range(i + 1, n):
        print(f"Sim('{sentences[i]}', '{sentences[j]}') = {sim_np[i, j]:.4f}")