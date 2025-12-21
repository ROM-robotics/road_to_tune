# Prompt Engineering: System, User, Assistant Roles

## မိတ်ဆက်

Prompt Engineering သည် LLM များထံမှ အကောင်းဆုံး output ရရှိရန် input (prompt) ကို ဒီဇိုင်းရေးဆွဲခြင်း ဖြစ်သည်။ Modern LLMs များတွင် သုံးမျိုးသော roles - **System**, **User**, **Assistant** ဟူ၍ ရှိပါသည်။

## Role-Based Prompting

### ၁. System Role (စနစ်အခန်းကဏ္ဍ)

**ရည်ရွယ်ချက်:**
System role သည် AI ၏ behavior, personality နှင့် context ကို သတ်မှတ်ပေးသည်။ ဤ instruction သည် conversation တစ်ခုလုံးအတွက် သက်ရောက်သည်။

**လက်တွေ့ဥပမာ (ROS2 Robot Assistant):**
```python
from openai import OpenAI

client = OpenAI()

# System prompt သတ်မှတ်ခြင်း
system_prompt = """သင်သည် ROS2 robotics expert တစ်ဦးဖြစ်သည်။ 
သင့်တာဝန်များ:
1. ROS2 commands များကို အတိအကျ ရှင်းပြပေးရန်
2. Nav2 navigation stack အသုံးပြုနည်းကို လမ်းညွှန်ပေးရန်
3. Safety considerations များကို အမြဲတမ်း ထည့်သွင်းစဉ်းစားရန်
4. Python နှင့် C++ code examples များ ပေးနိုင်ရန်

ဖြေဆိုပုံ: တိုတောင်းပြတ်သားပြီး technical accuracy အလေးပေးရမည်။"""

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "How do I make my robot move forward in ROS2?"}
    ]
)

print(response.choices[0].message.content)
```

**System Prompt ရေးသားရာတွင် သတ်မှတ်သင့်သည်များ:**
- AI ၏ expertise နှင့် role
- တုံ့ပြန်သင့်သော style (formal/casual, concise/detailed)
- လိုက်နာရမည့် rules နှင့ constraints
- Formatting preferences

---

### ၂. User Role (အသုံးပြုသူအခန်းကဏ္ဍ)

**ရည်ရွယ်ချက်:**
User role သည် လူသားအသုံးပြုသူ၏ မေးခွန်း သို့မဟုတ် instructions များကို ကိုယ်စားပြုသည်။

**လက်တွေ့ဥပမာများ:**

#### Example 1: Simple Question
```python
messages = [
    {
        "role": "system", 
        "content": "သင်သည် ROS2 tutor ဖြစ်သည်။ ရိုးရှင်းစွာ ရှင်းပြပါ။"
    },
    {
        "role": "user", 
        "content": "ROS2 node ဆိုတာ ဘာလဲ?"
    }
]
```

#### Example 2: Detailed Request
```python
messages = [
    {
        "role": "system", 
        "content": "သင်သည် robotics code reviewer ဖြစ်သည်။"
    },
    {
        "role": "user", 
        "content": """အောက်ပါ ROS2 node code ကို review လုပ်ပေးပါ:

'''python
import rclpy
from geometry_msgs.msg import Twist

def main():
    node = rclpy.create_node('simple_mover')
    pub = node.create_publisher(Twist, 'cmd_vel', 10)
    
    msg = Twist()
    msg.linear.x = 0.5
    pub.publish(msg)
'''

ပြဿနာများနှင့် တိုးတက်ရမည့်အချက်များ ညွှန်ပြပေးပါ။"""
    }
]
```

#### Example 3: Context-Rich Query
```python
user_message = """လက်ရှိအခြေအနေ: 
- Robot သည် warehouse environment တွင် ရှိနေသည်
- LIDAR sensor ပါရှိသည်
- Nav2 stack ကို install လုပ်ပြီးသည်

လုပ်ချင်သည်: Robot အား point A မှ point B သို့ autonomous navigation လုပ်စေချင်သည်။

မေးခွန်း: ဘယ် ROS2 commands များကို အစီအစဉ်တကျ run ရမလဲ?"""

messages = [
    {"role": "system", "content": "သင်သည် ROS2 Nav2 expert ဖြစ်သည်။"},
    {"role": "user", "content": user_message}
]
```

---

### ၃. Assistant Role (AI လက်ထောက်အခန်းကဏ္ဍ)

**ရည်ရွယ်ချက်:**
Assistant role သည် AI model ၏ response များကို ကိုယ်စားပြုသည်။ Multi-turn conversations တွင် previous responses များကို context အဖြစ် ထည့်သွင်းရန် အသုံးပြုသည်။

**လက်တွေ့ဥပမာ (Multi-turn Conversation):**
```python
conversation_history = [
    {
        "role": "system",
        "content": "သင်သည် ROS2 debugging assistant ဖြစ်သည်။"
    },
    {
        "role": "user",
        "content": "ကျွန်တော့် robot က cmd_vel commands များကို လက်ခံမနေပါ။"
    },
    {
        "role": "assistant",
        "content": """အောက်ပါအချက်များကို စစ်ဆေးပါ:

1. Topic name မှန်ကန်မှု:
   '''bash
   ros2 topic list | grep cmd_vel
   '''

2. Node များ running ဖြစ်မှု:
   '''bash
   ros2 node list
   '''

3. Topic info ကြည့်ရန်:
   '''bash
   ros2 topic info /cmd_vel
   '''

ဘယ် output များ ပြလာသလဲ?"""
    },
    {
        "role": "user",
        "content": "ros2 topic list မှာ /cmd_vel ပေါ်နေပါတယ်။ node list မှာလည်း /teleop_node ရှိပါတယ်။"
    },
    {
        "role": "assistant",
        "content": """ကောင်းပါသည်၊ topic နှင့် node ရှိနေပါပြီ။ နောက်ထပ် စစ်ရန်:

1. Message type မှန်ကန်မှု:
   '''bash
   ros2 topic info /cmd_vel -v
   '''

2. Manual test လုပ်ကြည့်ပါ:
   '''bash
   ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "linear: {x: 0.5}"
   '''

Robot လှုပ်လိုက်ပါသလား?"""
    },
    {
        "role": "user",
        "content": "Manual pub လုပ်တော့ robot လှုပ်ပါတယ်!"
    }
]

# နောက်ထပ် response ရယ်ခြင်း
response = client.chat.completions.create(
    model="gpt-4",
    messages=conversation_history
)

print(response.choices[0].message.content)
```

---

## Prompt Engineering Techniques

### ၁. Zero-Shot Prompting

**သဘောတရား:** Example မပေးဘဲ တိုက်ရိုက် မေးခြင်း

```python
prompt = "ROS2 node တစ်ခု ဖန်တီးတဲ့ Python code ရေးပေးပါ။"

messages = [
    {"role": "system", "content": "သင်သည် ROS2 developer ဖြစ်သည်။"},
    {"role": "user", "content": prompt}
]
```

---

### ၂. Few-Shot Prompting

**သဘောတရား:** Examples များပေးပြီး pattern သင်ကြားခြင်း

```python
few_shot_prompt = """အောက်ပါ examples များကို လေ့လာပါ:

Example 1:
Input: "Move forward 2 meters"
Output: {"linear": {"x": 0.5}, "angular": {"z": 0.0}, "duration": 4.0}

Example 2:
Input: "Turn right 90 degrees"
Output: {"linear": {"x": 0.0}, "angular": {"z": -1.57}, "duration": 1.0}

Example 3:
Input: "Stop immediately"
Output: {"linear": {"x": 0.0}, "angular": {"z": 0.0}, "duration": 0.0}

အခုဆိုရင် အောက်ပါကို convert လုပ်ပေးပါ:
Input: "Move backward 1 meter"
Output: """

messages = [
    {"role": "system", "content": "သင်သည် natural language to robot command converter ဖြစ်သည်။"},
    {"role": "user", "content": few_shot_prompt}
]
```

---

### ၃. Chain-of-Thought Prompting

**သဘောတရား:** AI အား step-by-step တွေးခိုင်းခြင်း

```python
cot_prompt = """Robot path planning problem တစ်ခုကို ဖြေရှင်းပါ:

ပြဿနာ: Robot သည် (0, 0) မှ (10, 10) သို့ သွားရမည်။ လမ်းတွင် (5, 5) တွင် obstacle ရှိသည်။

Step by step တွေးပြီး path plan လုပ်ပါ:
1. Current position နှင့် goal position ကို ခွဲခြားပါ
2. Obstacle ရှိမရှိ စစ်ဆေးပါ
3. Alternative paths များ စဉ်းစားပါ
4. အကောင်းဆုံး path ရွေးချယ်ပါ
5. Waypoints များ သတ်မှတ်ပါ"""

messages = [
    {"role": "system", "content": "သင်သည် robot path planner ဖြစ်သည်။ Step by step reasoning လုပ်ပါ။"},
    {"role": "user", "content": cot_prompt}
]
```

---

### ၄. Role Prompting

**သဘောတရား:** AI ကို specific expert role အနေနှင့် သတ်မှတ်ခြင်း

```python
# Role 1: Safety Inspector
safety_prompt = """သင်သည် robot safety inspector ဖြစ်သည်။ အောက်ပါ code ကို safety ရှုထောင့်က စစ်ဆေးပါ:

'''python
def emergency_stop(self):
    self.cmd_vel_pub.publish(Twist())  # Zero velocity
'''

Safety issues နှင့် improvements များ ညွှန်ပြပါ။"""
```

```python
# Role 2: Performance Optimizer
performance_prompt = """သင်သည် robotics performance optimizer ဖြစ်သည်။ အောက်ပါ navigation code ကို optimize လုပ်ရန် အကြံပြုပါ:

'''python
while not goal_reached:
    calculate_path()
    move_robot()
    time.sleep(0.1)
'''"""
```

---

## ROS2 အတွက် Prompt Templates

### Template 1: Command Generation
```python
def generate_ros2_command_prompt(task_description):
    return f"""Task: {task_description}

အောက်ပါ format ဖြင့် ROS2 command ထုတ်ပေးပါ:
1. လိုအပ်သော ROS2 packages
2. Terminal commands များ အစီအစဉ်တကျ
3. Expected output
4. Common errors နှင့် solutions

Command:"""
```

### Template 2: Code Review
```python
def code_review_prompt(code, language="python"):
    return f"""အောက်ပါ ROS2 {language} code ကို review လုပ်ပါ:

'''{language}
{code}
'''

စစ်ဆေးရမည့်အချက်များ:
1. ROS2 best practices လိုက်နာမှု
2. Error handling ရှိမရှိ
3. Resource cleanup (shutdown handling)
4. Thread safety
5. Performance considerations

Review:"""
```

### Template 3: Debugging Assistant
```python
def debug_prompt(error_message, context):
    return f"""Debug လုပ်ရန်:

Error Message:
'''
{error_message}
'''

Context:
{context}

အောက်ပါအတိုင်း ကူညီပေးပါ:
1. Error ဖြစ်ရသည့် ဖြစ်နိုင်ချေ အကြောင်းရင်းများ
2. စစ်ဆေးရမည့် အချက်များ (diagnostic commands)
3. ဖြေရှင်းနည်းများ priority အလိုက်
4. နောက်တစ်ခေါက် မဖြစ်အောင် ကာကွယ်နည်း

Solution:"""
```

---

## လေ့ကျင့်ခန်း

### Exercise 1: System Prompt Design
```python
# TODO: ROS2 navigation assistant အတွက် system prompt တစ်ခု ရေးပါ
# Requirements:
# - Nav2 stack expertise
# - Safety-first approach
# - မြန်မာလို ဖြေနိုင်ရမည်
# - Code examples ပေးနိုင်ရမည်

system_prompt = """
# သင့် prompt ကို ဤနေရာတွင် ရေးပါ
"""
```

### Exercise 2: Few-Shot Learning
```python
# TODO: Natural language robot commands ကို ROS2 Python code အဖြစ် ပြောင်းပေးမည့်
# few-shot prompt တစ်ခု ဖန်တီးပါ။ အနည်းဆုံး 3 examples ပေးပါ။

few_shot_examples = """
# သင့် examples များကို ဤနေရာတွင် ရေးပါ
"""
```

### Exercise 3: Multi-Turn Conversation
```python
# TODO: Robot debugging scenario အတွက် 5-turn conversation တစ်ခု ဒီဇိုင်းရေးဆွဲပါ
# User နှင့် Assistant အပြန်အလှန် ထိထိရောက်ရောက် ပြဿနာ ဖြေရှင်းနိုင်ရမည်

conversation = [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    # TODO: ဆက်ရေးပါ
]
```

---

## Best Practices

### ✅ လုပ်သင့်သည်များ
1. **Clear and Specific**: Prompt ကို တိကျရှင်းလင်းစေရန်
2. **Provide Context**: လိုအပ်သော background information ပေးရန်
3. **Use Examples**: Few-shot learning အတွက် quality examples များပေးရန်
4. **Define Format**: Expected output format ကို သတ်မှတ်ပေးရန်
5. **Iterate**: Prompt ကို test လုပ်ပြီး တိုးတက်အောင် ပြုပြင်ရန်

### ❌ ရှောင်သင့်သည်များ
1. **Vague Requests**: "Tell me about robots" လို မရှင်းလင်းသော prompts
2. **Too Complex**: တစ်ခါတည်း အလွန်များသော tasks များ တောင်းခြင်း
3. **No Context**: လုံလောက်သော information မပေးခြင်း
4. **Ignore Errors**: AI ၏ mistakes များကို feedback မပေးခြင်း

---

## အနှစ်ချုပ်

- **System Role**: AI ၏ behavior နှင့် expertise သတ်မှတ်သည်
- **User Role**: မေးခွန်းနှင့် instructions များ
- **Assistant Role**: AI responses များ (conversation history)
- **Prompt Engineering**: Effective prompts ရေးသားခြင်းသည် LLM အစွမ်းကို အပြည့်အဝ အသုံးချနိုင်စေသည်

နောက်သင်ခန်းစာတွင် Soft Fine-Tuning techniques များကို လေ့လာပါမည်။
