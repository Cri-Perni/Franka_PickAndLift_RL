# Franka Panda Robot RL Training 🤖

Deep Reinforcement Learning training environment for the Franka Panda robotic arm using PPO (Proximal Policy Optimization) and Genesis physics simulation.

## 📋 Project Overview

This project implements a complete RL pipeline for training a Franka Panda robot to perform reach-and-grasp tasks. The robot learns to locate, approach, grasp, and lift a cube placed at various positions (including 360° around the base).

### Key Features

- **Advanced PPO Configuration**: Optimized policy network with custom architecture (512-512-256 layers)
- **Generalized State Dependent Exploration (gSDE)**: Enables smooth, coherent exploration for continuous control
- **Parallel Training**: Multi-process vectorized environments (40 parallel simulations)
- **360° Manipulation**: Trained to handle objects in all directions around the robot base
- **Modular Reward System**: Separate components for approach, grasp, lift, and penalties
- **Genesis Simulation**: High-performance physics engine for robotics

## 🏗️ Project Structure

```
RL Final/
├── train.py              # Main training script with PPO configuration
├── play.py               # Script to visualize trained policies
├── fast_evaluate.py      # Quick evaluation script
├── env/
│   ├── franka_env.py     # Gymnasium environment implementation
│   ├── config.py         # Environment parameters and constants
│   ├── rewards.py        # Modular reward functions
│   └── utils.py          # Helper functions
├── model/                # Directory for saved models
├── logs/                 # Training checkpoints
└── tensorboard_logs/     # TensorBoard training metrics
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA support (recommended for training)
- Conda or venv for environment management

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/YOUR_USERNAME/franka-rl-training.git
cd franka-rl-training
```

2. **Create a Python environment**:
```bash
conda create -n genesis_rl python=3.10
conda activate genesis_rl
```

3. **Install dependencies**:
```bash
# Standard dependencies
pip install -r requirements.txt

# For CUDA support (GTX 1070 or similar):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 🎯 Training

### Start Training

Run the interactive training manager:

```bash
python train.py
```

You'll be prompted to choose:
1. **New Training** - Start from scratch
2. **Continue Training** - Resume from latest checkpoint
3. **Continue from Specific Model** - Load a specific .zip file

### Training Configuration

Key hyperparameters (in `train.py`):

- **Parallel Environments**: 40 (adjust based on CPU cores)
- **Total Timesteps**: 6,000,000
- **Learning Rate**: 3e-4
- **Batch Size**: 2048
- **N Steps**: 1024
- **Entropy Coefficient**: 0.01 (crucial for 360° exploration)
- **gSDE**: Enabled for smooth exploration

### Monitor Training

Launch TensorBoard to visualize training metrics:

```bash
tensorboard --logdir=./tensorboard_logs/
```

Navigate to `http://localhost:6006` to see real-time training graphs.

## 🎮 Evaluation & Visualization

### Play Trained Model

```bash
python play.py
```

### Quick Evaluation

```bash
python fast_evaluate.py
```

## 📊 Environment Details

### Observation Space

The agent receives a **36-dimensional observation** including:
- Robot joint positions (7 DoF)
- Robot joint velocities (7 DoF)
- Gripper finger positions (2)
- Gripper center position (3D)
- Cube position (3D)
- Relative distance vector (3D)
- Previous action (8)

### Action Space

**8-dimensional continuous action** space:
- Joint velocities for 7 arm joints: `[-1, 1]`
- Gripper control (open/close): `[-1, 1]`

### Reward Structure

Modular reward components:
- **Approach Reward**: Encourages moving toward the cube
- **Grasp Reward**: Incentivizes proper grasping configuration
- **Lift Reward**: Rewards lifting the cube above target height
- **Success Bonus**: Large reward for completing the task
- **Penalties**: Premature gripper closing, base collisions

## ⚙️ Customization

### Modify Training Parameters

Edit `train.py` to adjust:
- Number of parallel environments
- PPO hyperparameters
- Network architecture
- Entropy coefficient for exploration

### Modify Environment

Edit `env/config.py` to change:
- Cube spawn positions and difficulty
- Maximum episode steps
- Reward weights
- Physical parameters (friction, size, etc.)

## 🔧 Troubleshooting

### CUDA Out of Memory

Reduce batch size or number of parallel environments in `train.py`:
```python
num_envs = 20  # Reduce from 40
batch_size = 1024  # Reduce from 2048
```

### Import Errors

Ensure you're using the correct environment:
```bash
conda activate genesis_rl
which python  # Should point to genesis_rl environment
```

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{franka-rl-training,
  author = {Cristian Perniconi},
  title = {Franka Panda Robot RL Training with PPO},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/YOUR_USERNAME/franka-rl-training}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Stable-Baselines3** for the PPO implementation
- **Genesis** for high-performance physics simulation
- **OpenAI Gymnasium** for the RL environment interface

## 📬 Contact

For questions or issues, please open an issue on GitHub or contact [your-email@example.com].
