# Image Encoder + Language Decoder Architecture နားလည်ခြင်း

Vision-Language Models (VLMs) များသည် ပုံရိပ် (Image) နှင့် စာသား (Text) ကို ပေါင်းစပ်နားလည်နိုင်သော AI model များဖြစ်ကြသည်။ ၎င်းတို့၏ အဓိက တည်ဆောက်ပုံမှာ အပိုင်း (၃) ပိုင်း ပါဝင်သည်။

## 1. Image Encoder (Vision Part)
ပုံရိပ်များကို ကွန်ပျူတာနားလည်သော ကိန်းဂဏန်းများ (Embeddings) အဖြစ် ပြောင်းလဲပေးသည်။
- **Popular Models:** CLIP (Contrastive Language-Image Pre-training), ViT (Vision Transformer), SigLIP.
- **Function:** Input image ကို ယူပြီး visual feature vectors များ ထုတ်ပေးသည်။
- **Example:** $224 \times 224$ pixel ပုံတစ်ပုံကို $14 \times 14$ patch များခွဲပြီး vector sequence တစ်ခုအဖြစ် ပြောင်းလဲခြင်း။

## 2. Projection Layer (Connector/Adapter)
Image Encoder မှ ရလာသော visual features များကို Language Decoder နားလည်သော text embedding space သို့ ညှိပေးသော တံတားဖြစ်သည်။
- **Types:**
  - **Linear Layer:** ရိုးရှင်းသော matrix multiplication (ဥပမာ - LLaVA v1).
  - **MLP (Multi-Layer Perceptron):** ပိုမိုရှုပ်ထွေးသော features များကို map လုပ်ပေးသည် (ဥပမာ - LLaVA v1.5).
  - **Q-Former / Resampler:** Visual tokens အရေအတွက်ကို လျှော့ချပြီး အရေးကြီးသော အချက်အလက်များကိုသာ ယူသည် (ဥပမာ - BLIP-2, Flamingo).

## 3. Language Decoder (LLM Part)
Visual features နှင့် Text prompt ကို ပေါင်းစပ်ပြီး အဖြေထုတ်ပေးသည်။
- **Popular Models:** LLaMA, Vicuna, Qwen, Mistral.
- **Process:** 
  1. Image features များကို "visual tokens" အဖြစ် သတ်မှတ်သည်။
  2. User ၏ text prompt နှင့် visual tokens ကို concatenate လုပ်သည်။
  3. LLM က ၎င်းတို့ကို input အနေဖြင့် လက်ခံပြီး text response ကို autoregressive နည်းလမ်းဖြင့် ထုတ်ပေးသည်။

## Architecture Diagram
```mermaid
graph LR
    A[Input Image] --> B[Image Encoder\n(ViT/CLIP)]
    B --> C[Visual Features]
    C --> D[Projection Layer\n(Linear/MLP)]
    D --> E[Visual Embeddings]
    F[Text Prompt] --> G[Tokenizer]
    G --> H[Text Embeddings]
    E --> I[Concatenate]
    H --> I
    I --> J[Large Language Model\n(Decoder)]
    J --> K[Text Output]
```

## Summary
ဤ architecture သည် LLM ၏ စွမ်းရည် (reasoning, world knowledge) ကို အသုံးချပြီး ပုံရိပ်များကို နားလည်စေသည်။ Image encoder က "မျက်လုံး" အဖြစ်ဆောင်ရွက်ပြီး၊ LLM က "ဦးနှောက်" အဖြစ် ဆောင်ရွက်ကာ Projection layer က ၎င်းတို့နှစ်ခုကြားရှိ "ဘာသာပြန်" အဖြစ် ဆောင်ရွက်သည်။
