၁။ Warning Message:
You are using the default legacy behaviour... ဆိုတာက transformers library ရဲ့ version အပြောင်းအလဲကြောင့် ပေါ်လာတဲ့ သတိပေးချက် (Warning) သက်သက်ပါ။ Error မဟုတ်ပါဘူး။ ကုဒ်က ပုံမှန်အတိုင်း အလုပ်လုပ်နေပါတယ်။

၂။ Model Output:
Output: ROS2: Bewegen Sie den Roboter vorwärts
ဒီမှာ T5 model က "Move the robot forward" ဆိုတာကို ဂျာမန်ဘာသာ (German) နဲ့ "Bewegen Sie den Roboter vorwärts" လို့ ဘာသာပြန်ပေးလိုက်တာပါ။

ဘာကြောင့် ဒီလိုဖြစ်တာလဲ?

T5 model (pre-trained) က မူရင်းအတိုင်းဆိုရင် ဘာသာပြန်ခြင်း (Translation)၊ စာစီခြင်း (Summarization) စတာတွေကိုပဲ သင်ယူထားတာပါ။
"ROS2" command တွေကို သူ မသိသေးပါဘူး။
ဒါကြောင့် "translate English to ROS2" လို့ ခိုင်းလိုက်တဲ့အခါ သူနဲ့ အရင်းနှီးဆုံးဖြစ်တဲ့ "translate English to German" ပုံစံမျိုး မှားပြီး လုပ်ပေးလိုက်တာ ဖြစ်ပါတယ်။
ဖြေရှင်းရန်:
တကယ့် ROS2 command (ဥပမာ - /cmd_vel topic ပို့တာမျိုး) ထွက်လာဖို့ဆိုရင်၊ ROS2 command တွေပါတဲ့ Dataset

နဲ့ Model ကို Fine-tuning ပြန်လုပ်ပေးဖို့ လိုအပ်ပါတယ်။

