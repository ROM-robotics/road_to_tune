# ROS2 / Nav2 Use-Cases to Prompts

## မိတ်ဆက်

ROS2 နှင့် Nav2 use-cases များကို LLM training အတွက် prompts အဖြစ် ခွဲခြားသတ်မှတ်ခြင်းသည် robotics domain အတွက် specialized model များ ဖန်တီးရာတွင် အရေးကြီးသော အဆင့်ဖြစ်သည်။ ဤသင်ခန်းစာတွင် လက်တွေ့ use-cases များနှင့် အတူ prompt templates များကို လေ့လာမည်ဖြစ်သည်။

---

## Use-Case Categories

### ၁. Basic Robot Control
### ၂. Navigation & Path Planning  
### ၃. Sensor Data Processing
### ၄. Error Handling & Debugging
### ၅. System Configuration
### ၆. Multi-Robot Coordination

---

## ၁. Basic Robot Control Use-Cases

### Use-Case 1.1: Velocity Commands

**Real-world scenario:** Robot အား မတူညီသော အမြန်နှုန်းများဖြင့် လှုပ်ရှားစေခြင်း

```python
use_cases = [
    {
        "scenario": "Move robot forward at normal speed",
        "instruction": "Move the robot forward at 0.5 m/s",
        "output": "ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.5, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'",
        "context": "Basic forward motion command",
        "tags": ["movement", "linear", "basic"]
    },
    {
        "scenario": "Move robot backward slowly",
        "instruction": "Move the robot backward at 0.2 m/s",
        "output": "ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist '{linear: {x: -0.2, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'",
        "context": "Reverse motion for avoiding obstacles",
        "tags": ["movement", "linear", "reverse"]
    },
    {
        "scenario": "Rotate robot in place",
        "instruction": "Rotate the robot counterclockwise at 0.5 rad/s",
        "output": "ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.5}}'",
        "context": "In-place rotation for orientation adjustment",
        "tags": ["movement", "angular", "rotation"]
    },
    {
        "scenario": "Emergency stop",
        "instruction": "Stop the robot immediately",
        "output": "ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}'",
        "context": "Safety command to halt all motion",
        "tags": ["safety", "stop", "emergency"]
    },
    {
        "scenario": "Curve motion",
        "instruction": "Move forward while turning right",
        "output": "ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.3, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: -0.3}}'",
        "context": "Combined linear and angular motion for curved paths",
        "tags": ["movement", "curve", "combined"]
    }
]
```

### Prompt Template for Velocity Control

```python
velocity_template = """### Task: Robot Velocity Control

### Scenario:
{scenario}

### User Request:
{instruction}

### ROS2 Command:
{output}

### Explanation:
- linear.x: Forward/backward velocity (m/s, positive=forward)
- angular.z: Rotation velocity (rad/s, positive=counterclockwise)
- Use --once flag for single message publication

### Safety Notes:
{context}
"""
```

---

## ၂. Navigation & Path Planning Use-Cases

### Use-Case 2.1: Goal-Based Navigation

**Real-world scenario:** Robot အား တစ်နေရာမှ နောက်တစ်နေရာသို့ autonomous navigation လုပ်စေခြင်း

```python
navigation_use_cases = [
    {
        "scenario": "Navigate to charging station",
        "instruction": "Send the robot to the charging station at coordinates (5.0, 3.0)",
        "output": """ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose "{
  pose: {
    header: {frame_id: 'map'},
    pose: {
      position: {x: 5.0, y: 3.0, z: 0.0},
      orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
    }
  }
}" """,
        "context": "Autonomous navigation to specific map coordinates",
        "tags": ["navigation", "goal", "autonomous"],
        "prerequisites": ["Map loaded", "AMCL localized", "Nav2 running"]
    },
    {
        "scenario": "Navigate with specific orientation",
        "instruction": "Navigate to position (2.0, -1.0) facing east",
        "output": """ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose "{
  pose: {
    header: {frame_id: 'map'},
    pose: {
      position: {x: 2.0, y: -1.0, z: 0.0},
      orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
    }
  }
}" """,
        "context": "Goal with specific final orientation (quaternion)",
        "tags": ["navigation", "goal", "orientation"],
        "notes": "Orientation: East (0°) → w=1.0, z=0.0"
    },
    {
        "scenario": "Navigate through waypoints",
        "instruction": "Follow waypoints: (1,1) → (2,3) → (4,2)",
        "output": """ros2 action send_goal /follow_waypoints nav2_msgs/action/FollowWaypoints "{
  poses: [
    {
      header: {frame_id: 'map'},
      pose: {position: {x: 1.0, y: 1.0, z: 0.0}, orientation: {w: 1.0}}
    },
    {
      header: {frame_id: 'map'},
      pose: {position: {x: 2.0, y: 3.0, z: 0.0}, orientation: {w: 1.0}}
    },
    {
      header: {frame_id: 'map'},
      pose: {position: {x: 4.0, y: 2.0, z: 0.0}, orientation: {w: 1.0}}
    }
  ]
}" """,
        "context": "Sequential waypoint navigation",
        "tags": ["navigation", "waypoints", "sequential"]
    },
    {
        "scenario": "Cancel navigation",
        "instruction": "Cancel the current navigation goal",
        "output": "ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose --cancel-all",
        "context": "Stop autonomous navigation immediately",
        "tags": ["navigation", "cancel", "abort"]
    }
]
```

### Use-Case 2.2: Path Planning Queries

```python
path_planning_queries = [
    {
        "scenario": "Check if path exists",
        "instruction": "Can the robot navigate from current position to (10, 5)?",
        "output": """ros2 service call /compute_path_to_pose nav2_msgs/srv/ComputePathToPose "{
  goal: {
    header: {frame_id: 'map'},
    pose: {position: {x: 10.0, y: 5.0, z: 0.0}, orientation: {w: 1.0}}
  },
  use_start: false
}" """,
        "context": "Query path planner without executing navigation",
        "tags": ["planning", "query", "feasibility"]
    },
    {
        "scenario": "Get current robot pose",
        "instruction": "What is the robot's current position on the map?",
        "output": "ros2 topic echo --once /amcl_pose geometry_msgs/msg/PoseWithCovarianceStamped",
        "context": "Read localization estimate from AMCL",
        "tags": ["localization", "pose", "query"]
    }
]
```

---

## ၃. Sensor Data Processing Use-Cases

### Use-Case 3.1: LIDAR Data

```python
lidar_use_cases = [
    {
        "scenario": "Monitor LIDAR data",
        "instruction": "Show me the current laser scan data",
        "output": "ros2 topic echo /scan sensor_msgs/msg/LaserScan",
        "context": "Real-time LIDAR data stream",
        "tags": ["sensor", "lidar", "monitoring"]
    },
    {
        "scenario": "Check LIDAR frequency",
        "instruction": "What is the publishing rate of the laser scanner?",
        "output": "ros2 topic hz /scan",
        "context": "Verify sensor is publishing at expected rate",
        "tags": ["sensor", "lidar", "diagnostics"]
    },
    {
        "scenario": "Visualize LIDAR in RViz",
        "instruction": "Visualize the laser scan data",
        "output": "ros2 run rviz2 rviz2 -d $(ros2 pkg prefix nav2_bringup)/share/nav2_bringup/rviz/nav2_default_view.rviz",
        "context": "Launch RViz with navigation visualization",
        "tags": ["visualization", "lidar", "rviz"]
    }
]
```

### Use-Case 3.2: Camera & Odometry

```python
sensor_use_cases = [
    {
        "scenario": "Read odometry data",
        "instruction": "Show the robot's odometry information",
        "output": "ros2 topic echo /odom nav_msgs/msg/Odometry",
        "context": "Wheel encoder-based position estimate",
        "tags": ["sensor", "odometry", "monitoring"]
    },
    {
        "scenario": "Check transform tree",
        "instruction": "Display the TF tree",
        "output": "ros2 run tf2_tools view_frames",
        "context": "Visualize coordinate frame relationships",
        "tags": ["tf", "frames", "diagnostics"]
    },
    {
        "scenario": "Monitor battery status",
        "instruction": "What is the current battery level?",
        "output": "ros2 topic echo /battery_state sensor_msgs/msg/BatteryState",
        "context": "Monitor power supply status",
        "tags": ["sensor", "battery", "power"]
    }
]
```

---

## ၄. Error Handling & Debugging Use-Cases

### Use-Case 4.1: Common Errors

```python
error_scenarios = [
    {
        "error": "Transform timeout error",
        "symptom": "Failed to transform from 'base_link' to 'map'",
        "diagnosis": "Check TF tree and localization status",
        "solution_steps": [
            "ros2 run tf2_ros tf2_echo map base_link",
            "ros2 topic hz /tf",
            "ros2 service call /reinitialize_global_localization std_srvs/srv/Empty"
        ],
        "instruction": "Fix the transform timeout error between base_link and map",
        "output": """# Step 1: Verify TF is publishing
ros2 topic hz /tf

# Step 2: Check specific transform
ros2 run tf2_ros tf2_echo map base_link

# Step 3: If needed, reinitialize localization
ros2 service call /reinitialize_global_localization std_srvs/srv/Empty""",
        "tags": ["error", "tf", "localization"]
    },
    {
        "error": "Navigation failure",
        "symptom": "Robot stuck or cannot find path",
        "diagnosis": "Costmap or planner configuration issue",
        "solution_steps": [
            "ros2 topic echo /global_costmap/costmap",
            "ros2 param get /controller_server max_vel_x",
            "ros2 service call /global_costmap/clear_entirely_global_costmap nav2_msgs/srv/ClearEntireCostmap"
        ],
        "instruction": "Debug why the robot cannot find a path to the goal",
        "output": """# Step 1: Check if costmap is valid
ros2 topic echo --once /global_costmap/costmap nav_msgs/msg/OccupancyGrid

# Step 2: Clear costmap
ros2 service call /global_costmap/clear_entirely_global_costmap nav2_msgs/srv/ClearEntireCostmap

# Step 3: Verify planner is running
ros2 node info /planner_server""",
        "tags": ["error", "navigation", "debugging"]
    },
    {
        "error": "No map loaded",
        "symptom": "Map is empty or not published",
        "diagnosis": "Map server not running or map file missing",
        "solution_steps": [
            "ros2 run nav2_map_server map_server --ros-args -p yaml_filename:=/path/to/map.yaml",
            "ros2 service call /map_server/load_map nav2_msgs/srv/LoadMap"
        ],
        "instruction": "Load a map for navigation",
        "output": """# Method 1: Load map at startup
ros2 run nav2_map_server map_server --ros-args -p yaml_filename:=/path/to/your_map.yaml

# Method 2: Load map dynamically
ros2 service call /map_server/load_map nav2_msgs/srv/LoadMap "{map_url: '/path/to/your_map.yaml'}"

# Verify map is published
ros2 topic echo --once /map nav_msgs/msg/OccupancyGrid""",
        "tags": ["error", "map", "setup"]
    }
]
```

### Use-Case 4.2: Diagnostic Commands

```python
diagnostic_use_cases = [
    {
        "scenario": "List all active nodes",
        "instruction": "Show me all running ROS2 nodes",
        "output": "ros2 node list",
        "context": "Verify required nodes are running",
        "tags": ["diagnostics", "nodes", "status"]
    },
    {
        "scenario": "Check node details",
        "instruction": "Get information about the planner_server node",
        "output": "ros2 node info /planner_server",
        "context": "Shows subscriptions, publications, services, and actions",
        "tags": ["diagnostics", "nodes", "details"]
    },
    {
        "scenario": "List all parameters",
        "instruction": "Show all parameters of controller_server",
        "output": "ros2 param list /controller_server",
        "context": "View configurable parameters",
        "tags": ["diagnostics", "parameters", "config"]
    },
    {
        "scenario": "Monitor node computational load",
        "instruction": "Check CPU and memory usage of navigation nodes",
        "output": "ros2 run rqt_top rqt_top",
        "context": "Performance monitoring tool",
        "tags": ["diagnostics", "performance", "monitoring"]
    }
]
```

---

## ၅. System Configuration Use-Cases

### Use-Case 5.1: Parameter Configuration

```python
config_use_cases = [
    {
        "scenario": "Adjust max velocity",
        "instruction": "Set maximum linear velocity to 0.8 m/s",
        "output": "ros2 param set /controller_server max_vel_x 0.8",
        "context": "Runtime parameter adjustment",
        "tags": ["config", "velocity", "parameters"],
        "note": "Changes are temporary unless saved to param file"
    },
    {
        "scenario": "Change planner algorithm",
        "instruction": "Switch to Smac Planner 2D",
        "output": "ros2 param set /planner_server plugin_names '[\"GridBased\"]'\nros2 param set /planner_server GridBased.plugin 'nav2_smac_planner/SmacPlanner2D'",
        "context": "Change path planning algorithm",
        "tags": ["config", "planner", "algorithm"]
    },
    {
        "scenario": "Save current parameters",
        "instruction": "Save current navigation parameters to file",
        "output": "ros2 param dump /controller_server > controller_params.yaml",
        "context": "Export parameters for persistent configuration",
        "tags": ["config", "parameters", "save"]
    },
    {
        "scenario": "Load parameters from file",
        "instruction": "Load navigation parameters from YAML file",
        "output": "ros2 param load /controller_server controller_params.yaml",
        "context": "Import previously saved parameters",
        "tags": ["config", "parameters", "load"]
    }
]
```

### Use-Case 5.2: Launch Files

```python
launch_use_cases = [
    {
        "scenario": "Launch full navigation stack",
        "instruction": "Start Nav2 with default configuration",
        "output": "ros2 launch nav2_bringup navigation_launch.py",
        "context": "Brings up all Nav2 nodes",
        "tags": ["launch", "navigation", "full-stack"],
        "prerequisites": ["Map available", "Robot description loaded"]
    },
    {
        "scenario": "Launch SLAM Toolbox",
        "instruction": "Start SLAM mapping",
        "output": "ros2 launch slam_toolbox online_async_launch.py",
        "context": "Real-time SLAM mapping",
        "tags": ["launch", "slam", "mapping"]
    },
    {
        "scenario": "Launch with custom parameters",
        "instruction": "Start navigation with custom config file",
        "output": "ros2 launch nav2_bringup navigation_launch.py params_file:=/path/to/custom_params.yaml",
        "context": "Override default parameters",
        "tags": ["launch", "navigation", "custom"]
    }
]
```

---

## ၆. Multi-Robot Coordination Use-Cases

### Use-Case 6.1: Namespace Management

```python
multirobot_use_cases = [
    {
        "scenario": "Control robot_1",
        "instruction": "Move robot_1 forward",
        "output": "ros2 topic pub /robot_1/cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.5}}'",
        "context": "Use namespace prefix for multi-robot systems",
        "tags": ["multirobot", "namespace", "control"]
    },
    {
        "scenario": "Send robot_2 to goal",
        "instruction": "Navigate robot_2 to position (3, 4)",
        "output": """ros2 action send_goal /robot_2/navigate_to_pose nav2_msgs/action/NavigateToPose "{
  pose: {
    header: {frame_id: 'map'},
    pose: {position: {x: 3.0, y: 4.0, z: 0.0}, orientation: {w: 1.0}}
  }
}" """,
        "context": "Separate navigation goals for each robot",
        "tags": ["multirobot", "navigation", "coordination"]
    },
    {
        "scenario": "Monitor all robots",
        "instruction": "List topics for all robots",
        "output": "ros2 topic list | grep -E '/robot_[0-9]+/'",
        "context": "Filter topics by robot namespace",
        "tags": ["multirobot", "monitoring", "topics"]
    }
]
```

---

## Complete Prompt Dataset Generator

### Comprehensive Dataset Creator

```python
from typing import List, Dict
import json

class Nav2PromptDatasetGenerator:
    def __init__(self):
        self.templates = {
            "command": """### Task: {task_type}

### Scenario:
{scenario}

### User Request:
{instruction}

### ROS2 Command:
```bash
{output}
```

### Context:
{context}

### Tags: {tags}""",
            
            "troubleshooting": """### Problem: {error}

### Symptoms:
{symptom}

### Diagnostic Steps:
{diagnosis}

### Solution:
```bash
{output}
```

### Prevention:
{prevention}""",
            
            "explanation": """### Question:
{instruction}

### Answer:
{output}

### Technical Details:
{context}

### Related Topics: {tags}"""
        }
    
    def generate_dataset(self, use_cases: List[Dict], template_type: str = "command") -> List[Dict]:
        """Generate formatted prompts from use cases"""
        
        dataset = []
        template = self.templates[template_type]
        
        for uc in use_cases:
            formatted = template.format(**uc)
            
            dataset.append({
                "text": formatted,
                "metadata": {
                    "scenario": uc.get("scenario", ""),
                    "tags": uc.get("tags", []),
                    "difficulty": uc.get("difficulty", "medium")
                }
            })
        
        return dataset
    
    def save_dataset(self, dataset: List[Dict], filename: str):
        """Save to JSONL format"""
        with open(filename, 'w', encoding='utf-8') as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Saved {len(dataset)} examples to {filename}")

# အသုံးပြုနည်း
generator = Nav2PromptDatasetGenerator()

# Combine all use cases
all_use_cases = (
    use_cases + 
    navigation_use_cases + 
    path_planning_queries +
    lidar_use_cases +
    sensor_use_cases +
    diagnostic_use_cases +
    config_use_cases +
    launch_use_cases +
    multirobot_use_cases
)

# Generate dataset
dataset = generator.generate_dataset(all_use_cases, template_type="command")

# Save to file
generator.save_dataset(dataset, "nav2_training_dataset.jsonl")

print(f"Total examples generated: {len(dataset)}")
print(f"Categories: {len(set(uc.get('tags', [''])[0] for uc in all_use_cases))}")
```

---

## Prompt Engineering for ROS2

### Advanced Prompt Patterns

```python
# Pattern 1: Step-by-Step Reasoning
step_by_step_template = """### Problem:
{problem}

### Let's solve this step by step:

1. **Understand the requirement:**
   {step1}

2. **Identify necessary components:**
   {step2}

3. **Construct the command:**
   ```bash
   {command}
   ```

4. **Verify the result:**
   {verification}

### Final Command:
{output}"""

# Pattern 2: Error Recovery Chain
error_recovery_template = """### Issue Detected:
{error_description}

### Diagnosis Chain:
```bash
# Check 1: {check1_name}
{check1_command}

# Check 2: {check2_name}
{check2_command}

# Check 3: {check3_name}
{check3_command}
```

### Solution:
{solution}

### Verify Fix:
```bash
{verification_command}
```"""

# Pattern 3: Multi-Choice Selection
selection_template = """### Task:
{task}

### Available Options:

A) {option_a}
   Command: `{command_a}`
   Use when: {use_case_a}

B) {option_b}
   Command: `{command_b}`
   Use when: {use_case_b}

C) {option_c}
   Command: `{command_c}`
   Use when: {use_case_c}

### Recommendation:
{recommendation}

### Suggested Command:
```bash
{suggested_command}
```"""
```

---

## လေ့ကျင့်ခန်း

### Exercise 1: Create Custom Use-Cases
```python
# TODO: သင့် robot application အတွက် 10 use-cases ဖန်တီးပါ
# Include: scenario, instruction, output, context, tags

my_use_cases = [
    {
        "scenario": "...",
        "instruction": "...",
        "output": "...",
        "context": "...",
        "tags": [...]
    },
    # TODO: Add 9 more
]
```

### Exercise 2: Error Scenario Dataset
```python
# TODO: Common ROS2/Nav2 errors 5 ခု အတွက် troubleshooting prompts ဖန်တီးပါ

error_dataset = [
    {
        "error": "...",
        "symptom": "...",
        "diagnosis": "...",
        "solution_steps": [...],
        "prevention": "..."
    },
    # TODO: Add more
]
```

### Exercise 3: Multi-Turn Conversations
```python
# TODO: Navigation task တစ်ခုအတွက် multi-turn conversation dataset ဖန်တီးပါ
# Example: User asks question → Assistant answers → User follows up → etc.

conversation = [
    {"role": "user", "content": "How do I start navigation?"},
    {"role": "assistant", "content": "First, ensure you have..."},
    # TODO: Continue conversation (at least 5 turns)
]
```

---

## Dataset Statistics & Validation

### Quality Metrics

```python
def analyze_dataset(dataset: List[Dict]) -> Dict:
    """Dataset quality analysis"""
    
    stats = {
        "total_examples": len(dataset),
        "avg_instruction_length": 0,
        "avg_output_length": 0,
        "tag_distribution": {},
        "scenario_types": {},
        "command_types": {}
    }
    
    for item in dataset:
        # Calculate lengths
        inst_len = len(item.get("instruction", ""))
        out_len = len(item.get("output", ""))
        
        stats["avg_instruction_length"] += inst_len
        stats["avg_output_length"] += out_len
        
        # Count tags
        for tag in item.get("tags", []):
            stats["tag_distribution"][tag] = stats["tag_distribution"].get(tag, 0) + 1
        
        # Identify command types
        if "ros2 topic" in item.get("output", ""):
            stats["command_types"]["topic"] = stats["command_types"].get("topic", 0) + 1
        elif "ros2 action" in item.get("output", ""):
            stats["command_types"]["action"] = stats["command_types"].get("action", 0) + 1
        elif "ros2 service" in item.get("output", ""):
            stats["command_types"]["service"] = stats["command_types"].get("service", 0) + 1
    
    stats["avg_instruction_length"] /= len(dataset)
    stats["avg_output_length"] /= len(dataset)
    
    return stats

# Analyze
stats = analyze_dataset(all_use_cases)
print(json.dumps(stats, indent=2))
```

---

## အနှစ်ချုပ်

### Key Takeaways

1. **Categorize Use-Cases**: Control, Navigation, Sensors, Errors, Config, Multi-robot
2. **Include Context**: Scenario, prerequisites, expected outcomes
3. **Add Metadata**: Tags, difficulty, related topics
4. **Provide Examples**: Real commands with explanations
5. **Cover Edge Cases**: Errors, recovery, troubleshooting

### Dataset Best Practices

✅ **လုပ်သင့်သည်များ:**
- Real-world scenarios အခြေခံ
- Clear, executable commands
- Context နှင့် explanation ပါဝင်ရမည်
- Multiple difficulty levels
- Error handling ထည့်သွင်းရမည်

❌ **ရှောင်သင့်သည်များ:**
- Theoretical-only examples
- Incomplete commands
- Missing context
- No error scenarios
- Outdated ROS2 syntax

---

## နိဂုံး

ROS2/Nav2 use-cases များကို prompts အဖြစ် သတ်မှတ်ခြင်းသည်:
- **Structured Learning Data** ဖန်တီးပေးသည်
- **Real-World Applicability** ရှိစေသည်
- **Scalable Training** လုပ်နိုင်စေသည်
- **Domain Expertise** transfer လုပ်နိုင်စေသည်

ဤ week 1 သင်ခန်းစာများပြီးဆုံးပါပြီ။ သင်သည် LLM basics မှ ROS2 prompt engineering အထိ လမ်းကြောင်းတစ်ခုလုံး လေ့လာပြီးဖြစ်ပါသည်။

**နောက်ထပ် လေ့လာရန်များ:**
- Week 2: Advanced Fine-Tuning Techniques
- Week 3: Model Evaluation & Deployment
- Week 4: Production Systems & Monitoring

Happy Learning! 🤖🚀
