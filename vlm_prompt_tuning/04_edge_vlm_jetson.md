# Jetson Orin / Edge Device များတွင် Lightweight VLM သုံးခြင်း

Robotics တွင် Cloud ပေါ်တင်ပြီး process လုပ်ခြင်းသည် latency နှင့် privacy ပြဿနာများရှိနိုင်သဖြင့်၊ Robot ပေါ်တွင် (On-device / Edge) တိုက်ရိုက် run နိုင်သော Lightweight VLM များ လိုအပ်သည်။ NVIDIA Jetson Orin ကဲ့သို့ device များသည် ဤအတွက် အသင့်တော်ဆုံးဖြစ်သည်။

## 1. Suitable Lightweight VLMs
Edge device များတွင် VRAM ကန့်သတ်ချက်ရှိသဖြင့် parameter နည်းသော model များကို ရွေးချယ်ရမည်။

- **NanoLLaVA / LLaVA-Phi:** ( < 3B parameters) အလွန်ပေါ့ပါးပြီး မြန်ဆန်သည်။
- **Qwen-VL-Chat-Int4:** 4-bit quantization ပြုလုပ်ထားသော version ဖြစ်ပြီး memory သက်သာသည်။
- **BakLLaVA:** Mistral ပေါ်အခြေခံထားပြီး 7B size ရှိသော်လည်း quantization ဖြင့် run နိုင်သည်။
- **Vila / LLaVA-NeXT:** ပိုမိုကောင်းမွန်သော architecture ရှိသော model အသစ်များ။

## 2. Optimization Techniques
Jetson Orin (8GB/16GB/32GB/64GB RAM) ပေါ်တွင် run ရန် အောက်ပါနည်းလမ်းများ သုံးရသည်။

### A. Quantization (4-bit / 8-bit)
Model weights များကို precision လျှော့ချခြင်းဖြင့် memory usage ကို 2-4 ဆ လျှော့ချနိုင်သည်။
- **Tools:** `bitsandbytes`, `AWQ`, `GPTQ`.
- **Example:** FP16 (14GB) -> 4-bit (4-5GB) for a 7B model.

### B. TensorRT-LLM
NVIDIA ၏ library ဖြစ်ပြီး Jetson GPU ပေါ်တွင် inference speed ကို အမြင့်ဆုံးရအောင် optimize လုပ်ပေးသည်။
- **Benefit:** PyTorch ထက် 2x-4x ပိုမြန်နိုင်သည်။

### C. Vision Encoder Optimization
Image encoder (CLIP/ViT) ကိုလည်း TensorRT engine အဖြစ်ပြောင်းလဲပြီး run နိုင်သည်။

## 3. Running on Jetson Orin (Example)
`jetson-containers` သို့မဟုတ် `dusty-nv` ၏ container များကို အသုံးပြုခြင်းသည် အလွယ်ကူဆုံးဖြစ်သည်။

### Using NanoLLaVA
```bash
# Pull the container
docker pull dustynv/l4t-text-generation:r36.2.0

# Run container
docker run -it --rm --runtime=nvidia --network=host \
    dustynv/l4t-text-generation:r36.2.0 \
    python3 -m nanollava.chat --model_path "googlenet/nanollava"
```

### Using VILA (Visual Language Model) with 4-bit
```bash
python3 -m llava.serve.cli \
    --model-path efficient-large-model/VILA-2.7b \
    --load-4bit
```

## 4. Performance Considerations
- **Jetson Orin Nano (8GB):** 
  - Recommended: < 3B models (e.g., NanoLLaVA, Phi-3-Vision).
  - 7B models 4-bit ဖြင့် run ရန် ခက်ခဲနိုင်သည်။
- **Jetson Orin NX (16GB):** 
  - Recommended: 7B models (4-bit quantized).
  - Qwen-VL-Chat-Int4 ကောင်းစွာ run နိုင်သည်။
- **Jetson AGX Orin (32GB/64GB):** 
  - 13B models သို့မဟုတ် 7B models (FP16) ကို သက်တောင့်သက်သာ run နိုင်သည်။

## Summary
Edge VLM များသည် Robot များကို အင်တာနက်မလိုဘဲ ပတ်ဝန်းကျင်ကို နားလည်နိုင်စွမ်း ပေးသည်။ Model ရွေးချယ်ရာတွင် Hardware ၏ VRAM ပမာဏနှင့် ချိန်ညှိရန် အရေးကြီးဆုံးဖြစ်သည်။
