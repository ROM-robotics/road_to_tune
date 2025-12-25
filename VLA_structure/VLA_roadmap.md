<!-- ...existing code... -->

# Roadmap: Depth Camera နှင့် Qwen/Qwen2.5-VL-3B-Instruct ကို Differential Controller (နှစ်ဘီး) အတွက် အသုံးပြုခြင်း

## ၁။ Hardware Setup
- သင့် robot တွင် အသုံးပြုနိုင်သော depth camera (ဥပမာ Intel RealSense, ZED) တစ်ခု ရွေးချယ်ပါ။
- Depth camera ကို robot ၏ onboard computer (ဥပမာ Raspberry Pi, Jetson, x86) နှင့် ချိတ်ဆက်ပါ။
- နှစ်ဘီး differential drive motors နှင့် motor drivers ကို တပ်ဆင်ပါ။

## ၂။ software environment
- ROS2 (Robot Operating System 2) ကို တပ်ဆင်ပါ။
- Qwen/Qwen2.5-VL-3B-Instruct model ကို Python environment တွင် အသုံးပြုရန် ပြင်ဆင်ပါ။
- လိုအပ်သော package များ တပ်ဆင်ပါ။
  - Depth camera ROS2 drivers
  - Qwen/Qwen2.5-VL-3B-Instruct model dependencies (transformers, torch, စသည်)

## ၃။ Depth Camera ကို ROS2 တွင် ပေါင်းစပ်ခြင်း
- Depth camera node ကို ROS2 တွင် run ပါ။
- Depth image နှင့် point cloud topics များ ထုတ်ပေးနေကြောင်း စစ်ဆေးပါ။
- လိုအပ်လျှင် camera ကို ချိန်ညှိပါ။

## ၄။ Qwen/Qwen2.5-VL-3B-Instruct ကို ပေါင်းစပ်ခြင်း
- Qwen/Qwen2.5-VL-3B-Instruct model ကို download ပြုလုပ်ပြီး ပြင်ဆင်ပါ။
- Depth image များကို model သို့ ပို့ရန် script တစ်ခု ပြုလုပ်ပါ။
- Scene နားလည်မှုနှင့် navigation အတွက် prompt များ ပြုလုပ်ပါ။

## ၅။ Perception Pipeline
- ROS2 တွင် depth camera topics များ subscribe လုပ်ပါ။
- Depth data ကို preprocess (ဥပမာ အတားအဆီး detection, free space mapping) ပြုလုပ်ပါ။
- Qwen model ကို အသုံးပြု၍ scene ကို နားလည်ပြီး navigation command များ ထုတ်ပါ။

## ၆။ Differential Controller ကို တည်ဆောက်ခြင်း
- ROS2 differential drive controller ကို အသုံးပြုပါ (သို့မဟုတ် အသစ်ရေးပါ။)
- Qwen model output ကို velocity command (`cmd_vel` topic) သို့ mapping ပြုလုပ်ပါ။
- ရိုးရိုး forward, turn, stop command များ စမ်းသပ်ပါ။

## ၇။ Closed-Loop Navigation
- Depth camera မှ feedback ကို obstacle avoidance အတွက် အသုံးပြုပါ။
- Qwen model ကို high-level decision making (ဥပမာ "target သို့သွားပါ", "obstacle ကိုရှောင်ပါ") အတွက် အသုံးပြုပါ။
- Controller parameter များကို tune လုပ်ပါ။

## ၈။ စမ်းသပ်ခြင်းနှင့် အကဲဖြတ်ခြင်း
- Real-world deployment မပြုလုပ်မီ simulation (Gazebo, RViz) တွင် စမ်းသပ်ပါ။
- Perception နှင့် control pipeline ကို အမှန်တကယ်ပတ်ဝန်းကျင်တွင် စစ်ဆေးပါ။
- စမ်းသပ်ရလဒ်အပေါ် မူတည်၍ တိုးတက်အောင် ပြင်ဆင်ပါ။

## ၉။ Documentation နှင့် Maintenance
- Setup, code, အသုံးပြုနည်းများကို မှတ်တမ်းတင်ပါ။
- ဆော့ဖ်ဝဲကို လိုအပ်သလို update ပြုလုပ်ပါ။

---

**ကိုးကားချက်များ:**
- ROS2 Documentation: https://docs.ros.org/en/foxy/index.html
- Qwen Model: https://github.com/QwenLM/Qwen-VL
- Depth Camera ROS2 Drivers: [Intel RealSense](https://github.com/IntelRealSense/realsense-ros), [ZED](https://www.stereolabs.com/docs/ros2/)