# Robotics Perception: Object, Obstacle, Scene Understanding

VLM များကို Robotics Perception တွင် အသုံးပြုခြင်းသည် သမားရိုးကျ Object Detection model များထက် ပိုမိုနက်ရှိုင်းသော နားလည်မှု (Semantic Understanding) ကို ပေးစွမ်းနိုင်သည်။

## 1. Object Detection & Recognition (Open-Vocabulary)
YOLO ကဲ့သို့ model များသည် train ထားသော class များကိုသာ သိရှိနိုင်သော်လည်း၊ VLM များသည် မည်သည့်အရာဝတ္ထုကိုမဆို စာသားဖြင့် ဖော်ပြနိုင်သည်။

- **Task:** ပုံထဲရှိ အရာဝတ္ထုများကို ရှာဖွေခြင်း။
- **Prompt:** "Detect all 'cups' and 'bottles' in the image and provide bounding boxes."
- **Advantage:** Training data မလိုဘဲ အရာဝတ္ထုအသစ်များကို ရှာဖွေနိုင်ခြင်း (Zero-shot detection)။

## 2. Obstacle Avoidance & Navigability Analysis
Robot သွားလာမည့် လမ်းကြောင်းတွင် အတားအဆီးများကို ခွဲခြမ်းစိတ်ဖြာခြင်း။

- **Task:** လမ်းကြောင်းရှင်းမရှင်း စစ်ဆေးခြင်း။
- **Prompt:** "Is the path ahead clear for a robot to move forward? Identify any blocking obstacles."
- **Response Example:** "The path is blocked by a cardboard box in the center. There is a clear space on the left side."
- **Application:** Nav2 costmap တွင် semantic layer အနေဖြင့် ထည့်သွင်းအသုံးပြုနိုင်သည်။

## 3. Scene Understanding & Context
ပတ်ဝန်းကျင်၏ အခြေအနေကို နားလည်ခြင်း (ဥပမာ - မီးဖိုချောင်၊ ရုံးခန်း၊ လူစည်ကားသောနေရာ)။

- **Task:** လက်ရှိရောက်နေသော နေရာနှင့် အခြေအနေကို သုံးသပ်ခြင်း။
- **Prompt:** "Describe the current environment. Is it safe for a cleaning robot to operate?"
- **Response Example:** "This is a living room with children playing on the floor. It is not safe to operate currently due to dynamic obstacles."

## 4. Actionable Insights (Reasoning)
မြင်ရသောအရာပေါ်မူတည်ပြီး ဘာလုပ်သင့်သည်ကို ဆုံးဖြတ်ခြင်း။

- **Scenario:** Robot လက်တံဖြင့် ပစ္စည်းကောက်ခြင်း။
- **Prompt:** "I want to pick up the mug. Which part of the mug should I grasp?"
- **Response:** "You should grasp the handle of the mug, which is facing to the right."

## Example Workflow in ROS2
1. **Camera Node:** ပုံရိပ်ဖမ်းယူသည်။
2. **VLM Node:** ပုံရိပ်နှင့် Prompt ("Check for obstacles") ကို လက်ခံရယူသည်။
3. **Inference:** VLM မှ "Chair is blocking the path" ဟု အဖြေပေးသည်။
4. **Navigation Node:** Robot ကို ရပ်တန့်စေသည် သို့မဟုတ် လမ်းကြောင်းပြောင်းစေသည်။

## Summary
VLM များသည် Robotics တွင် "မြင်ရုံသာမက နားလည်ခြင်း" (Seeing and Understanding) ကို ပေးစွမ်းနိုင်ပြီး၊ ရှုပ်ထွေးသော ပတ်ဝန်းကျင်များတွင် ဆုံးဖြတ်ချက်ချရန် အလွန်အသုံးဝင်သည်။
