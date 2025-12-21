# Prompt Engineering: System, User, Assistant Roles

## မိတ်ဆက်

Prompt Engineering သည် LLM များထံမှ အကောင်းဆုံး output ရရှိရန် input (prompt) ကို ဒီဇိုင်းရေးဆွဲခြင်း ဖြစ်သည်။ Modern LLMs များတွင် သုံးမျိုးသော roles - **System**, **User**, **Assistant** ဟူ၍ ရှိပါသည်။

## Role-Based Prompting

### ၁. System Role (စနစ်အခန်းကဏ္ဍ)

**ရည်ရွယ်ချက်:**
System role သည် AI ၏ behavior, personality နှင့် context ကို သတ်မှတ်ပေးသည်။ ဤ instruction သည် conversation တစ်ခုလုံးအတွက် သက်ရောက်သည်။

**လက်တွေ့ဥပမာ (ROS2 Robot Assistant):**
```python
# check B01_system_role.py
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
        "content": "You are a ROS2 tutor. Explain simply."
    },
    {
        "role": "user", 
        "content": "what is ROS2 node."
    }
]
```

#### Example 2: Detailed Request
```python
messages = [
    {
        "role": "system", 
        "content": "You are a robotics code reviewer."
    },
    {
        "role": "user", 
        "content": """Please review the ROS2 node code below.:

'''python
import rclpy
from geometry_msgs.msg import Twist

def main():
    node = rclpy.create_node('simple_mover')
    pub = node.create_publisher(Twist, 'cmd_vel', 10)
    
    msg = Twist();
    msg.linear.x = 0.5
    pub.publish(msg)
'''

please find syntax errors and logic errors. """
    }
]
```

#### Example 3: Context-Rich Query
```python
user_message = """Current status: 
- Robot is in a warehouse environment
- LIDAR sensor is present
- Nav2 stack is installed

What I want to do: I want the robot to navigate autonomously from point A to point B.

Question: Which ROS2 commands should I run in sequence?"""

messages = [
    {"role": "system", "content": "You are Ros2 humble Nav2 expert."},
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
        "content": "You are a ROS2 debugging assistant."
    },
    {
        "role": "user",
        "content": "My robot is not accepting cmd_vel commands."
    },
    {
        "role": "assistant",
        "content": """Please check the following:

1. Topic name correctness:
   '''bash
   ros2 topic list | grep cmd_vel
   '''

2. Node running :
   '''bash
   ros2 node list
   '''

3. Check Topic info:
   '''bash
   ros2 topic info /cmd_vel
   '''

What output is displayed??"""
    },
    {
        "role": "user",
        "content": "The ros2 topic list shows /cmd_vel. The node list also shows /teleop_node."
    },
    {
        "role": "assistant",
        "content": """Okay, topic and node are there. More to check:

1. Message type correctness:
   '''bash
   ros2 topic info /cmd_vel -v
   '''

2. Try a manual test:
   '''bash
   ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "linear: {x: 0.5}"
   '''

Did the robot move?"""
    },
    {
        "role": "user",
        "content": "When I do a manual pub, the robot moves!"
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
prompt = "Write Python code that creates a ROS2 node."

messages = [
    {"role": "system", "content": "you are ROS2 developer."},
    {"role": "user", "content": prompt}
]
```

---

### ၂. Few-Shot Prompting

**သဘောတရား:** Examples များပေးပြီး pattern သင်ကြားခြင်း

```python
few_shot_prompt = """Study the examples below.:

Example 1:
Input: "Move forward 2 meters"
Output: {"linear": {"x": 0.5}, "angular": {"z": 0.0}, "duration": 4.0}

Example 2:
Input: "Turn right 90 degrees"
Output: {"linear": {"x": 0.0}, "angular": {"z": -1.57}, "duration": 1.0}

Example 3:
Input: "Stop immediately"
Output: {"linear": {"x": 0.0}, "angular": {"z": 0.0}, "duration": 0.0}

Now convert the following:
Input: "Move backward 8 meter"
Output: """

messages = [
    {"role": "system", "content": "You are a natural language to ROS2 humble robot command converter."},
    {"role": "user", "content": few_shot_prompt}
]
```

---

### ၃. Chain-of-Thought Prompting

**သဘောတရား:** AI အား step-by-step တွေးခိုင်းခြင်း

```python
cot_prompt = """Solve a robot path planning problem:

Problem: The robot must go from (0, 0) to (10, 10). There is an obstacle at (5, 5) on the way.

Plan the path by thinking step by step:

1. Distinguish between the current position and the goal position
2. Check for obstacles
3. Consider alternative paths
4. Choose the best path
5. Set waypoints"""

messages = [
    {"role": "system", "content": "You are a robot path planner. Do step by step reasoning."},
    {"role": "user", "content": cot_prompt}
]
```

---

### ၄. Role Prompting

**သဘောတရား:** AI ကို specific expert role အနေနှင့် သတ်မှတ်ခြင်း

```python
# Role 1: Safety Inspector
safety_prompt = """I am a robot safety inspector. Please check the following code from a safety perspective:

'''python
def emergency_stop(self):
    self.cmd_vel_pub.publish(Twist())  # Zero velocity
'''

Point Safety issues and improvements"""
```

```python
# Role 2: Performance Optimizer
performance_prompt = """You are a robotics performance optimizer. Please suggest optimizing the following navigation code:

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
