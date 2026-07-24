# CSCN8020 Assignment 3: Deep Q-Network Control of the Unitree G1 Left Elbow
## Deep Q-Network (DQN) Left Elbow Joint Position Controller

### Student Details
* **Student Name:** Chao-Chung Liu
* **Student ID:** 9067679
* **Course:** CSCN8020 - Reinforcement Learning
* **Instructor:** Prof. Enrique Espinosa
* **Repository GitHub URL:** https://github.com/caatat741213/CSCN8020_Assignment-3.git

---

### Project Description
This repository contains the complete implementation and empirical analysis of a Deep Q-Network (DQN) agent designed to control the single-joint left elbow (`left_elbow_joint`) of the Unitree G1 humanoid robot in a simulated MuJoCo environment.

Building upon the Low-Level Proportional-Derivative (PD) joint-target modulator and gravity/bias-force compensation developed during the **Unitree MuJoCo G1 Primer Workshop**, this project replaces the hand-written rule-based policy with a model-free, value-based reinforcement learning controller. 

The agent learns to command discrete target adjustment actions ($\Delta \theta \in \{-0.05, 0, +0.05\}$ rad) based on the continuous robot state and joint goals. The final policy generalizes across multiple target angles, achieves 100% success rate, stabilizes the joint near target angles, and significantly outperforms the rule-based heuristic controller in stabilization time and steady-state error.

---

### Environment Setup Instructions
To set up the simulation and reinforcement learning environment on Windows 11 using WSL 2 (Ubuntu 24.04):

1. **Install system dependencies (Prerequisites for WSL):**
   Update your package list and install the required system libraries for git, Python virtual environment, and MuJoCo's OpenGL/GLFW rendering (essential for running simulation rendering and preventing package import errors):
   ```bash
   sudo apt update && sudo apt install -y python3-venv python3-dev build-essential git cmake ninja-build libglfw3  libglfw3-dev  libgl1-mesa-dev  libegl1-mesa-dev  libxinerama-dev  libxcursor-dev  libxi-dev  libxrandr-dev

   ```

2. **Activate virtual environment & Install dependencies:**
   Make sure you are in the workspace root:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

3. **Clone and Integrate External Unitree MuJoCo dependency:**
   We use the official Unitree MuJoCo repository as an external dependency at a specific commit hash for compatibility:
   ```bash
   git clone https://github.com/unitreerobotics/unitree_mujoco.git external/unitree_mujoco
   git -C external/unitree_mujoco checkout ae6a8403e272733e9996ef59990880330496177f
   ```

4. **Verify the environment configuration:**
   Run the environment test script to ensure registration and compatibility:
   ```bash
   PYTHONPATH=src python src/test_g1_elbow_env.py
   ```

---

### DQN Implementation Details
The DQN agent and training workflow are fully written in **PyTorch** and follow a modular structure:

1. **Q-Network (`q_network.py`):**
   * **Input (4 Dimensions):** `[current angle, current velocity, goal angle, goal - current angle]`
   * **Hidden Layers:** MLP with 2 hidden layers of 64 units each, applying `ReLU` activations.
   * **Output (3 Dimensions):** Action Q-values corresponding to the three discrete actions: `[0 (DECREASE), 1 (HOLD), 2 (INCREASE)]`. The final layer does **not** apply Softmax, as Q-values are unconstrained.

2. **Replay Buffer (`replay_buffer.py`):**
   * Bounded experience replay using a `collections.deque` with a capacity of `50,000` transitions.
   * Samples random mini-batches of size `64` to break temporal correlation.
   * Device-aware: Automatically transfers sampled states, actions, rewards, next states, and terminal masks to the selected execution device (default is CPU).

3. **DQN Agent (`agent.py`):**
   * Instantiates the online Q-network ($Q$) and a target Q-network ($\hat{Q}$).
   * Performs $\epsilon$-greedy exploration selection (`select_action`) during training, and greedy action selection ($\epsilon=0.0$) during evaluation.
   * Standardizes temporal difference learning using Huber Loss (`nn.SmoothL1Loss()`) for training stability, optimized with the Adam optimizer (learning rate $\eta = 0.001$).
   * Synchronizes the target network parameters to the online network every `250` optimization steps.
   * Implements gradient clipping at a maximum norm of `1.0` to prevent exploding gradients.
   * **Bootstrapping Logic:** Properly handles Gymnasium termination and truncation signals. Only true terminal states (when joint error is within tolerance for a consecutive duration, indicating success) set `terminated=True` and mask bootstrapping. Time-limit truncation (`truncated=True` at 150 steps) does **not** mask bootstrapping, allowing correct value estimation.

4. **Training Script (`train_dqn.py`):**
   * Runs the training loop headlessly on CPU to ensure portability.
   * Supports command-line execution for Configurations A-E.
   * Seeds all random number generators (Python `random`, `numpy`, `torch`, `gymnasium`) to ensure 100% reproducibility.
   * Logs training metrics (`Reward`, `Success`, `Steps`, `Final_Absolute_Error`, `Epsilon`, `Loss`, `Wall_Clock_Time`) to structured CSV files.

---

### Precise Installation & Execution Commands

All scripts should be executed from the repository root:

* **Module Smoke Test:**
  Confirm replay insertion, batch sampling, network forward pass, and backpropagation step are working correctly:
  ```bash
  PYTHONPATH=src python src/dqn/smoke_test.py
  ```

* **Training DQN Configurations (A to E):**
  Train a specific configuration (a: Baseline, b: Faster Decay, c_linear: Linear Decay, d_fast_target: Fast Target Update, e_small_buffer: Small Buffer) for a chosen number of episodes (default is 700):
  ```bash
  # Train Configuration A (Baseline)
  PYTHONPATH=src python src/dqn/train_dqn.py --config a --episodes 700
  
  # Train Configuration B (Faster Decay - Selected Official Model)
  PYTHONPATH=src python src/dqn/train_dqn.py --config b --episodes 700
  
  # Train Configuration C (Linear Decay - Empirical Winner)
  PYTHONPATH=src python src/dqn/train_dqn.py --config c_linear --episodes 700
  ```

* **Evaluating a Trained Model Checkpoint:**
  Evaluate a checkpoint greedily ($\epsilon = 0.0$) over the 20-episode benchmark suite:
  ```bash
  # Evaluate Selected DQN model
  PYTHONPATH=src python src/dqn/evaluate_dqn.py --checkpoint models/selected_dqn.pt
  ```

* **Evaluating the Rule-Based Baseline:**
  Evaluate the heuristic rule-based controller on the identical 20-episode benchmark:
  ```bash
  PYTHONPATH=src python src/dqn/evaluate_baseline.py
  ```

* **Comparing and Selecting Winner Automatically:**
  Read and analyze evaluation logs, copy the winning checkpoint to `models/selected_dqn.pt`, and output a summary report:
  ```bash
  PYTHONPATH=src python src/dqn/analyze_and_select_model.py
  ```

* **Plotting Training Curves & Comparisons:**
  Generate paper-ready high-resolution metrics plots from the training results:
  ```bash
  PYTHONPATH=src python src/dqn/plot_results.py
  ```

* **Visualizing the Learned Policy in MuJoCo Viewer:**
  Render the trained policy dynamically in the MuJoCo viewer over the 4 benchmark target angles:
  ```bash
  PYTHONPATH=src python src/dqn/render_dqn_policy.py
  ```

* **Running the Jupyter Notebook:**
  Launch Jupyter Notebook to view and run the interactive codebase:
  ```bash
  jupyter notebook CSCN8020_Assignment_3.ipynb
  ```

---

### Discussion & Answers

Below are precise and concise answers to the core assignment discussion questions:

1. **Which policy is more sample efficient?**
   * **Rule-based baseline:** Requires **0 training samples** (heuristic mathematical law), making it the overall most sample-efficient.
   * **Among DQN configurations (Sample Efficiency Comparison):** **Config B (Faster Decay, 0.985)** is the most sample-efficient learning policy. It converges much faster than Config A and Config C, reaching the 80% rolling success rate threshold at **Episode 60** compared to Episode 100+ for other configurations. By decreasing exploration early, it focuses on exploitation and saves training time and resources.

2. **Which policy is more stable near the goal?**
   * **DQN policy** is significantly more stable.
   * **Stability & Control Behavior:** The rule-based controller is purely proportional and lacks velocity awareness. Upon reaching the goal, high kinetic energy causes overshoot, forcing it to cycle actions (oscillating between increase/decrease) and introducing steady-state jitter.
   * The **DQN agent** reads the joint angular velocity $\dot{\theta}_t$ in its observation state. It learns to utilize **active braking** (applying reverse torque when error decreases while velocity is high) and selects the `HOLD` action (Action 1) to damp movements. This effectively eliminates overshoot and micro-oscillations.

3. **Does the DQN generalize across all four target angles?**
   * **Yes.** The DQN agent generalized perfectly, achieving a **100% success rate** (20/20 episodes) across all four required benchmark targets ($-0.8, -0.4, +0.4, +0.8$ rad).
   * Generalization is achieved because the relative error $e_t = \theta_g - \theta_t$ is explicitly included in the observation vector. The neural network learns a target-agnostic error minimization policy.

4. **Does the DQN learn to use HOLD appropriately?**
   * **Yes.** The DQN agent learns to use `HOLD` (Action 1) to stabilize the elbow joint near the goal. Rather than continuous action-jittering, it chooses `HOLD` when the error and velocity approach zero, maintaining position and preventing steady-state error.

5. **Are there signs of oscillation or unnecessary target changes?**
   * **Rule-based policy:** Displays micro-oscillations near the target due to a lack of dynamic damping.
   * **Config D (Fast Target Update):** Exhibits severe loss and reward training oscillations because the target network updates too frequently (every 50 steps), propagating function approximation errors.
   * **Config E (Small Buffer):** Displays higher variance in training rewards due to catastrophic forgetting and temporal overfitting.
   * **Selected DQN (Config B & C):** Exhibits minimal oscillation and is highly stable at the target.

6. **Why might a hand-written policy (like PID) outperform a learned policy in this simple task?**
   * **Zero Training Time:** A hand-written PID controller works immediately with 100% sample efficiency, requiring zero training episodes.
   * **No Function Approximation Error:** PID is based on deterministic physical formulas and has no approximation error or neural network generalization gaps.
   * **Continuous Precision:** PID outputs continuous torque adjustments, bypassing the precision limitations of DQN's discrete action step size ($\Delta\theta = 0.05$ rad).

---

### Empirical Performance Comparison

#### 20-Episode Benchmark Evaluation (All Configurations)
| Configuration | Epsilon Decay | Successes/20 | Success Rate | Mean Reward | Mean Steps | Mean Angle Error (rad) |
|---|---|---|---|---|---|---|
| **Config A (Baseline)** | 0.995 (Exp) | 20/20 | 100.0% | 13.2623 | 19.75 | 0.005197 |
| **Config B (Faster Decay)** | 0.985 (Exp) | 20/20 | 100.0% | 13.3026 | 19.50 | 0.010608 |
| **Config C (Linear Decay)** | Linear (500 ep) | 20/20 | 100.0% | **13.3429** | **19.50** | 0.006779 |
| **Config D (Fast Target)** | 0.995 (Exp) | 20/20 | 100.0% | 13.1241 | 20.25 | 0.008227 |
| **Config E (Small Buffer)** | 0.995 (Exp) | 20/20 | 100.0% | 13.1991 | 19.50 | 0.008837 |

#### Rule-Based Baseline vs. Selected DQN (Official Config B Winner)
| Metric | Rule-based Policy | Selected DQN (Config B) |
|---|---|---|
| **Successes/20** | 20/20 | 20/20 |
| **Success Rate** | 100.0% | 100.0% |
| **Mean Cumulative Reward** | 12.8666 | **13.3026** (Increase of +0.436) |
| **Mean Episode Length (Steps)** | 24.00 | **19.50** (Reduction of -4.50 steps) |
| **Mean Final Angle Error (rad)** | 0.012209 | **0.010608** (Reduction of -0.0016 rad) |
| **Main Qualitative Behaviour** | Proportional heuristic. Target changes immediately and stays at limit. Can lead to static error or sluggishness in joint due to lack of velocity prediction. | Learned value-driven policy. Dynamically selects actions to build torque. Learns to hold target angle near goal, reducing error and oscillation. |

---

### Repository File Structure

```text
CSCN8020_Assignment 3/
├── assets/
│   └── g1_fixed_base/
│       ├── scene_29dof_fixed_base.xml       # Fixed-base MuJoCo scene definition
│       └── scene_29dof_fixed_base_viewer.xml# Scene definition configured for viewer rendering
├── external/
│   └── unitree_mujoco/                      # Cloned official Unitree MuJoCo simulator repository
├── models/
│   ├── config_a.pt                          # Model checkpoint for Config A
│   ├── config_b.pt                          # Model checkpoint for Config B
│   ├── config_c_linear.pt                   # Model checkpoint for Config C
│   ├── config_d_fast_target.pt              # Model checkpoint for Config D
│   ├── config_e_small_buffer.pt             # Model checkpoint for Config E
│   └── selected_dqn.pt                      # Official Selected DQN checkpoint (copied from Config B)
├── report/
│   └── DQN_Assignment_Report.pdf            # Student's final technical report
├── results/
│   ├── config_a/
│   │   ├── eval_results.csv                 # Configuration A 20-episode evaluation metrics
│   │   └── training_log.csv                 # Configuration A training log (700 episodes)
│   ├── config_b/
│   │   ├── eval_results.csv                 # Configuration B 20-episode evaluation metrics
│   │   └── training_log.csv                 # Configuration B training log (700 episodes)
│   ├── config_c_linear/
│   │   ├── eval_results.csv                 # Configuration C 20-episode evaluation metrics
│   │   └── training_log.csv                 # Configuration C training log (700 episodes)
│   ├── config_d_fast_target/
│   │   ├── eval_results.csv                 # Configuration D 20-episode evaluation metrics
│   │   └── training_log.csv                 # Configuration D training log (700 episodes)
│   ├── config_e_small_buffer/
│   │   ├── eval_results.csv                 # Configuration E 20-episode evaluation metrics
│   │   └── training_log.csv                 # Configuration E training log (700 episodes)
│   ├── plots/                               # Metric visualization charts (loss, rewards, success rates)
│   ├── model_selection_report.md            # Generated comparison and selection report
│   ├── rule_based_baseline.csv              # Rule-based policy evaluation logs
│   └── single_joint_control.csv             # Manual PD joint control verification log
├── src/
│   ├── dqn/
│   │   ├── __init__.py                      # Package initialization
│   │   ├── agent.py                         # DQNAgent class handling learning step logic
│   │   ├── analyze_and_select_model.py      # Automated model selector script
│   │   ├── evaluate_baseline.py             # Baseline evaluation script
│   │   ├── evaluate_dqn.py                  # Trained model evaluation script
│   │   ├── plot_results.py                  # High-resolution results plotter
│   │   ├── q_network.py                     # 4x64x64x3 MLP QNetwork class
│   │   ├── render_dqn_policy.py             # Gym GUI renderer script using saved model
│   │   ├── replay_buffer.py                 # Experience Replay buffer class
│   │   ├── smoke_test.py                    # DQN component smoke-test script
│   │   └── train_dqn.py                     # Headless DQN training runner
│   ├── g1_rl/
│   │   ├── __init__.py                      # Environment module initialization
│   │   └── g1_elbow_env.py                  # Gymnasium custom G1ElbowTargetEnv wrapper
│   ├── control_single_joint.py              # PD joint-tracking demonstration script
│   ├── create_fixed_base_g1.py              # XML parser generating fixed-base humanoid model
│   ├── demo_g1_elbow_env.py                 # Interactive workspace camera & command debugger
│   ├── inspect_g1_model.py                  # MuJoCo G1 structural inspection tool
│   └── test_g1_elbow_env.py                 # Core environment registration & determinism test
├── CSCN8020_Assignment_3.ipynb              # Executable Jupyter Notebook with pre-saved training & evaluation logs
├── README.md                                # Setup, installation, execution commands, and discussion (this file)
├── requirements-lock.txt                    # Detailed package lockfile
├── requirements.txt                         # Key Python dependencies
└── .gitignore                               # Exclude rules (venv, cache, local raw simulation data)
```

---

### Verification Environment

Final verification was executed under the following hardware and software specifications:
* **Operating System:** Windows 11 Home/Pro, running **WSL 2 (Ubuntu 24.04 LTS)**
* **Python Version:** **Python 3.12.x** (within virtual environment `.venv`)
* **Hardware Profile:** Multi-core Intel/AMD x86_64 CPU (training is strictly CPU-compatible, CPU execution time $\approx 45 \text{s}$ to $60 \text{s}$ per 700-episode config)
* **Graphics Server (WSLg):** Enabled (Mesa software rendering or GPU passthrough used for Gymnasium GUI `render_dqn_policy.py`)
* **Primary Python Dependencies:**
  * `torch >= 2.0.0`
  * `gymnasium == 1.3.0`
  * `mujoco == 3.10.0`
  * `numpy == 2.5.1`
  * `pandas`
  * `matplotlib`
