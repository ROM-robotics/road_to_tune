
# LLM Soft Fine-Tuning & Robotics Roadmap

ဒီ document က **Soft Fine-Tuning (Prompt Tuning)** ကို စပြီး  
Robotics (ROS2 / Nav2 / VLM) အထိ တိုးချဲ့နိုင်ဖို့ **အဆင့်လိုက် Learning Roadmap** ဖြစ်ပါတယ်။

---

## Week 1 – Foundations (Soft Fine-Tuning Basics)

- LLM basics: Encoder / Decoder / Transformer concept နားလည်ခြင်း  
- Prompt Engineering (system, user, assistant roles)  
- Soft Fine-Tuning (Prompt Tuning, P-Tuning) သဘောတရား  
- HuggingFace Transformers & PEFT library အသုံးပြုနည်း  
- Dataset formatting (instruction + output single text format)  
- ROS2 / Nav2 use-case များကို prompt အဖြစ် ခွဲခြားသတ်မှတ်ခြင်း  

---

## Week 2 – Practical Prompt Tuning

- Qwen / LLaMA model ကို local environment မှာ run လုပ်ခြင်း  
- Virtual tokens (20–50) နဲ့ Prompt Tuning training  
- CPU / GPU environment အတွက် training config ချိန်ညှိခြင်း  
- Output format consistency (YAML, XML, launch files) ကို evaluate လုပ်ခြင်း  
- ROS Nav2 waypoint YAML, parameter explanation dataset သင်ခြင်း  

---

## Week 3 – Integration & Deployment

- Soft Fine-Tuned model ကို inference pipeline ထဲထည့်ခြင်း  
- RAG (Retrieval Augmented Generation) နဲ့ ROS docs ချိတ်ဆက်ခြင်း  
- Local ROSGPT node အဖြစ် deploy လုပ်ခြင်း  
- Prompt + RAG + Soft FT ကို combine လုပ်ပြီး hallucination လျော့ချခြင်း  
- Performance, latency, accuracy evaluation  

---

## Advanced Topics (Next Step)

### 1. LoRA (Low-Rank Adaptation)

- Base LLM weight မပြောင်းဘဲ adapter weight သင်ခြင်း  
- Reasoning, planning, code generation skill တိုးချဲ့နိုင်  
- ROS2 action planning, C++ code generation အတွက် သင့်တော်  

### 2. VLM Prompt Tuning (Vision + Language)

- Image encoder + Language decoder architecture နားလည်ခြင်း  
- Visual prompt (image + text) conditioning  
- Robotics perception: object, obstacle, scene understanding  
- Jetson Orin / edge device များတွင် lightweight VLM သုံးခြင်း  

### 3. Nav2 Behavior Planning with LLM

- Natural language → Nav2 action mapping  
- NavigateToPose / NavigateThroughPoses planning  
- Behavior Tree (BT XML) auto-generation  
- Mission-level planner (human command → robot behavior)  
- AMR / delivery robot use-cases  

---

ဒီ roadmap အတိုင်း လေ့လာပြီး လက်တွေ့လုပ်နိုင်ရင်  
**Local LLM + ROS2 + Nav2** ကို production level အထိ တိုးချဲ့နိုင်ပါတယ် 🚀
