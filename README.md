# Drone Navigation with PPO

## Project Overview
This project uses Proximal Policy Optimization (PPO) reinforcement learning to train a drone to navigate through obstacles and reach a target location.

## Demo
<video src="https://github.com/sohamkanhe/Autonomous-Drone-Navigation-using-Reinforcement-Learning/blob/main/drone_rl.mp4" width="100%" controls></video>

## What is PPO?
PPO is a reinforcement learning algorithm that teaches the drone by trial and error. The drone tries actions, gets rewards or penalties, and gradually learns what works best.

## Problem Setup
### State (what the drone sees)
- Relative position to target (X, Y, Z)
- 10 ray sensors detecting obstacles (0 = clear, 1 = obstacle)

### Actions (what the drone can do)
- Forward / Backward
- Left / Right  
- Up / Down

Each action moves the drone 0.2 meters.

### Reward System (how the drone learns)
- **Moving toward target:** small positive reward
- **Reaching target:** +100 points
- **Hitting obstacle or ground:** -10 points, episode ends
- **Each time step:** -0.1 points (encourages speed)

## Files
- `drone_rlurdf.py`: Main training/testing script
- `cf2_minimal.urdf`: Drone model
- `environment_dae.urdf`: 3D obstacle environment
- `cf2.dae`: Drone mesh file
- `uploads-files-4359726-city+2.dae`: Environment mesh
- `drone_model.zip`: Saved trained model
- `logs/`: Training data for TensorBoard
- `checkpoints/`: Backup models during training

## How to Use

### Train a new model (headless - faster)
```bash
python drone_rlurdf.py --mode train
```

### Train with visual simulation
```bash
python drone_rlurdf.py --mode train --gui
```

### Test a trained model
```bash
python drone_rlurdf.py --mode test
```

### View training progress in browser
```bash
tensorboard --logdir ./logs
```
Then open [http://localhost:6006](http://localhost:6006)

## Test Mode Controls
Before testing, position the target with these keys:
- **I**: Forward
- **K**: Backward
- **J**: Left
- **L**: Right
- **U**: Up
- **O**: Down
- **Space**: Start testing

## Expected Results

### Starting training
- Episode reward: -299
- Episode length: 500 steps
- Success rate: 0%

### After training
- Episode reward: +87.9
- Episode length: 34 steps  
- Success rate: 100%
- Collision rate: 0%

## Training Process
1. Drone starts at random position.
2. Takes action based on current knowledge.
3. Receives reward and new observation.
4. Updates its strategy after 2048 steps.
5. Repeats until it learns optimal behavior.

### Training Progress
- **Early:** Random movement, negative rewards.
- **Middle:** Learns to avoid obstacles.
- **Late:** Direct path to target, no collisions.

## Collision Rules
- Touching environment obstacles = collision
- Falling below 0.5 meters height = collision
- Reaching within 0.3 meters of target = success

## Notes
- Training without GUI is 2-3 times faster.
- Model auto-saves every 5000 steps.
- Press `Ctrl+C` to stop training (model saves).
- All logs saved to `logs` folder.
