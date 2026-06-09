import pybullet as p
import pybullet_data
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
import os
import torch
import time
import argparse
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import subprocess
import threading

DRONE_MIN_HEIGHT = 0.5
DRONE_MAX_HEIGHT = 5.0
STEP_SIZE = 0.2
RAY_LENGTH = 1.5

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        self.episode_collisions = []
        self.episode_count = 0
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.current_episode_success = 0
        self.current_episode_collision = 0

    def _on_step(self) -> bool:
        # Accumulate reward for current episode
        self.current_episode_reward += self.locals['rewards'][0]
        self.current_episode_length += 1
        
        # Check for episode end
        done = self.locals['dones'][0]
        if done:
            self.episode_count += 1
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.episode_successes.append(self.current_episode_success)
            self.episode_collisions.append(self.current_episode_collision)
            
            # Log to TensorBoard
            self.logger.record("train/episode_reward", self.current_episode_reward)
            self.logger.record("train/episode_length", self.current_episode_length)
            self.logger.record("train/success_rate", self.current_episode_success)
            self.logger.record("train/collision_rate", self.current_episode_collision)
            
            # Reset for next episode
            self.current_episode_reward = 0
            self.current_episode_length = 0
            self.current_episode_success = 0
            self.current_episode_collision = 0
        
        # Update success/collision flags from environment
        info = self.locals['infos'][0]
        if 'success' in info and info['success']:
            self.current_episode_success = 1
        if 'collision' in info and info['collision']:
            self.current_episode_collision = 1
            
        return True

class DroneEnv(gym.Env):
    def __init__(self, render=False, fixed_target_pos=None):
        super(DroneEnv, self).__init__()
        self.fixed_target_pos = fixed_target_pos

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)

        self.render_mode = render
        self.physics_client = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)

        self.max_steps = 500
        self.current_step = 0
        self.collision_count = 0
        self.success_count = 0

        self.setup_scene()

    def setup_scene(self):
        p.resetSimulation()

        if self.render_mode:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.resetDebugVisualizerCamera(
                cameraDistance=10,
                cameraYaw=0,
                cameraPitch=-45,
                cameraTargetPosition=[0, 0, 2]
            )

        p.loadURDF("plane.urdf")
        env_path = os.path.join(os.path.dirname(__file__), "environment_dae.urdf")
        p.loadURDF(env_path, basePosition=[0, 0, 0])

        self.drone = p.loadURDF("cf2_minimal.urdf", basePosition=[0, 0, DRONE_MIN_HEIGHT])
        self.target = None
        self.place_target()

    def place_target(self):
        if self.fixed_target_pos is not None:
            pos = self.fixed_target_pos
        else:
            while True:
                x = np.random.uniform(-5, 5)
                y = np.random.uniform(-5, 5)
                z = np.random.uniform(DRONE_MIN_HEIGHT, DRONE_MAX_HEIGHT)
                if not self.check_point_collision(x, y, z):
                    break
            pos = [x, y, z]

        if self.target:
            p.resetBasePositionAndOrientation(self.target, pos, [0, 0, 0, 1])
        else:
            self.target = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=0.2),
                baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.2, rgbaColor=[0, 1, 0, 1]),
                basePosition=pos
            )

    def check_point_collision(self, x, y, z):
        ray_start = [x, y, z]
        ray_end = [x, y, z - 0.1]
        result = p.rayTest(ray_start, ray_end)[0]
        return result[0] != -1

    def check_drone_collision(self):
        collisions = p.getContactPoints(bodyA=self.drone)
        return len(collisions) > 0

    def get_observation(self):
        drone_pos, _ = p.getBasePositionAndOrientation(self.drone)
        target_pos, _ = p.getBasePositionAndOrientation(self.target)

        directions = [
            (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
            (1, 1, 0), (-1, 1, 0), (1, -1, 0), (-1, -1, 0),
            (0, 0, 1), (0, 0, -1)
        ]

        ray_results = []
        for dx, dy, dz in directions:
            start = [drone_pos[0], drone_pos[1], drone_pos[2]]
            end = [start[0] + dx * RAY_LENGTH,
                   start[1] + dy * RAY_LENGTH,
                   start[2] + dz * RAY_LENGTH]
            result = p.rayTest(start, end)[0]
            ray_results.append(1.0 if result[0] != -1 else 0.0)

        rel_target = [target_pos[i] - drone_pos[i] for i in range(3)]
        return np.array(rel_target + ray_results, dtype=np.float32)

    def step(self, action):
        if self.render_mode:
            p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)

        drone_pos, _ = p.getBasePositionAndOrientation(self.drone)
        new_pos = list(drone_pos)

        if action == 0: new_pos[1] += STEP_SIZE  # forward
        elif action == 1: new_pos[1] -= STEP_SIZE  # back
        elif action == 2: new_pos[0] -= STEP_SIZE  # left
        elif action == 3: new_pos[0] += STEP_SIZE  # right
        elif action == 4: new_pos[2] += STEP_SIZE  # up
        elif action == 5: new_pos[2] -= STEP_SIZE  # down

        new_pos[2] = max(DRONE_MIN_HEIGHT, new_pos[2])
        p.resetBasePositionAndOrientation(self.drone, new_pos, [0, 0, 0, 1])

        obs = self.get_observation()
        target_pos, _ = p.getBasePositionAndOrientation(self.target)

        dist = np.linalg.norm(np.array(new_pos) - np.array(target_pos))
        reward = -dist * 0.1

        done = False
        collision_occurred = False
        success_occurred = False
        
        if self.check_drone_collision():
            reward -= 10
            done = True
            self.collision_count += 1
            collision_occurred = True
        if dist < 0.3:
            reward += 100
            done = True
            self.success_count += 1
            success_occurred = True

        reward -= 0.1
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True

        if self.render_mode:
            time.sleep(1/60)

        # Additional info for logging
        info = {
            'distance': dist,
            'position': new_pos,
            'target': target_pos,
            'collision': collision_occurred,
            'success': success_occurred
        }
        
        return obs, reward, done, info

    def reset(self):
        self.current_step = 0
        p.resetBasePositionAndOrientation(self.drone, [0, 0, DRONE_MIN_HEIGHT], [0, 0, 0, 1])
        self.place_target()
        return self.get_observation()

    def close(self):
        p.disconnect()

def start_tensorboard(log_dir):
    """Start TensorBoard in a separate thread"""
    def run_tensorboard():
        subprocess.run(["tensorboard", "--logdir", log_dir, "--port", "6006"])
    
    print("Starting TensorBoard on http://localhost:6006/")
    threading.Thread(target=run_tensorboard, daemon=True).start()

def train(use_gui=False):
    try:
        # Create environment
        env = DroneEnv(render=use_gui)
        
        # Create log directory
        log_dir = "./logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # Start TensorBoard
        start_tensorboard(log_dir)

        # Create callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=5000, save_path="./checkpoints", name_prefix="drone_model"
        )
        tensorboard_callback = TensorboardCallback()
        
        # Load or create model
        if os.path.exists("drone_model.zip"):
            model = PPO.load("drone_model", env=env, tensorboard_log=log_dir)
            print("Loaded existing model")
        else:
            model = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                device="cuda" if torch.cuda.is_available() else "cpu",
                tensorboard_log=log_dir
            )

        # Train the model
        model.learn(
            total_timesteps=1000000, 
            callback=[checkpoint_callback, tensorboard_callback],
            tb_log_name="PPO"
        )
        
        # Save the model
        model.save("drone_model")
        print("Training complete.")
        print(f"Total episodes: {tensorboard_callback.episode_count}")
        print(f"Success rate: {sum(tensorboard_callback.episode_successes)/tensorboard_callback.episode_count:.2f}")
        print(f"Collision rate: {sum(tensorboard_callback.episode_collisions)/tensorboard_callback.episode_count:.2f}")
        
    except KeyboardInterrupt:
        model.save("drone_model")
        print("Training interrupted and saved.")
    finally:
        if 'env' in locals():
            env.close()

def test_model(camera_mode='follow'):
    STEP = 0.2
    MOVE_KEYS = {
        ord('i'): 'i',
        ord('k'): 'k',
        ord('j'): 'j',
        ord('l'): 'l',
        ord('u'): 'u',
        ord('o'): 'o'
    }

    print("\nUse keys to position target:")
    print("  I/K: move forward/backward (Y axis)")
    print("  J/L: move left/right (X axis)")
    print("  U/O: move up/down (Z axis)")
    print("Press SPACEBAR to start testing...")

    env = DroneEnv(render=True)
    model = PPO.load("drone_model")

    target = env.target
    selected_pos, _ = p.getBasePositionAndOrientation(target)
    selected_pos = list(selected_pos)
    start_testing = False

    while not start_testing:
        keys = p.getKeyboardEvents()
        
        for key_code in keys:
            if key_code in MOVE_KEYS:
                direction = MOVE_KEYS[key_code]
                step = STEP * 0.1
                if direction == 'i': selected_pos[1] += step
                elif direction == 'k': selected_pos[1] -= step
                elif direction == 'j': selected_pos[0] -= step
                elif direction == 'l': selected_pos[0] += step
                elif direction == 'u': selected_pos[2] += step
                elif direction == 'o': selected_pos[2] = max(DRONE_MIN_HEIGHT, selected_pos[2] - step)

        p.resetBasePositionAndOrientation(target, selected_pos, [0, 0, 0, 1])

        if ord(' ') in keys:
            start_testing = True

        p.stepSimulation()
        time.sleep(1/240)

    print("\nStarting testing...")
    
    # Set up camera based on mode
    if camera_mode == 'static':
        drone_pos, _ = p.getBasePositionAndOrientation(env.drone)
        p.resetDebugVisualizerCamera(
            cameraDistance=10,
            cameraYaw=45,
            cameraPitch=-45,
            cameraTargetPosition=drone_pos
        )
        print("Using static camera view")
    else:
        print("Using follow camera view")
    
    collision_count = 0
    success_count = 0
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(10):
        obs = env.reset()
        p.resetBasePositionAndOrientation(target, selected_pos, [0, 0, 0, 1])
        
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            if camera_mode == 'follow':
                drone_pos, _ = p.getBasePositionAndOrientation(env.drone)
                p.resetDebugVisualizerCamera(
                    cameraDistance=3.5,
                    cameraYaw=45,
                    cameraPitch=-30,
                    cameraTargetPosition=drone_pos
                )
            
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            
            p.stepSimulation()
            time.sleep(1/60)
            
        if info['collision']:
            collision_count += 1
        if info['success']:
            success_count += 1
            
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        print(f"Test {episode+1}: Steps: {steps}, Reward: {total_reward:.2f}, " +
              f"Distance: {info['distance']:.2f}, " +
              f"Success: {info['success']}, Collision: {info['collision']}")

    env.close()
    
    # Generate test performance plots
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(episode_rewards, 'o-')
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    
    plt.subplot(2, 2, 2)
    plt.plot(episode_lengths, 'o-')
    plt.title("Episode Lengths")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    
    plt.subplot(2, 2, 3)
    plt.bar(['Success', 'Collision'], [success_count, collision_count])
    plt.title("Test Outcomes")
    plt.ylabel("Count")
    
    plt.tight_layout()
    plt.savefig("./logs/test_performance.png")
    print("Saved test performance plot to logs/test_performance.png")

def plot_training_results():
    log_dir = "./logs"
    if not os.path.exists(log_dir):
        print("Log directory not found!")
        return
        
    # Find the latest event file
    event_files = []
    for root, dirs, files in os.walk(log_dir):
        for f in files:
            if "events.out.tfevents" in f:
                event_files.append(os.path.join(root, f))
    
    if not event_files:
        print("No training data found!")
        return
        
    latest_file = sorted(event_files, key=os.path.getmtime)[-1]
    
    ea = event_accumulator.EventAccumulator(latest_file)
    ea.Reload()
    
    if not ea.Tags()['scalars']:
        print("No training data found in event file!")
        return
        
    plt.figure(figsize=(15, 10))
    
    # Reward plot
    if 'train/episode_reward' in ea.Tags()['scalars']:
        reward_data = ea.Scalars('train/episode_reward')
        steps = [x.step for x in reward_data]
        rewards = [x.value for x in reward_data]
        
        plt.subplot(2, 2, 1)
        plt.plot(steps, rewards)
        plt.title("Training Rewards")
        plt.xlabel("Timesteps")
        plt.ylabel("Mean Episode Reward")
    
    # Length plot
    if 'train/episode_length' in ea.Tags()['scalars']:
        length_data = ea.Scalars('train/episode_length')
        steps = [x.step for x in length_data]
        lengths = [x.value for x in length_data]
        
        plt.subplot(2, 2, 2)
        plt.plot(steps, lengths)
        plt.title("Episode Lengths")
        plt.xlabel("Timesteps")
        plt.ylabel("Mean Episode Length")
    
    # Success rate
    if 'train/success_rate' in ea.Tags()['scalars']:
        success_data = ea.Scalars('train/success_rate')
        steps = [x.step for x in success_data]
        rates = [x.value for x in success_data]
        
        plt.subplot(2, 2, 3)
        plt.plot(steps, rates)
        plt.title("Success Rate")
        plt.xlabel("Timesteps")
        plt.ylabel("Success Rate")
    
    # Collision rate
    if 'train/collision_rate' in ea.Tags()['scalars']:
        collision_data = ea.Scalars('train/collision_rate')
        steps = [x.step for x in collision_data]
        rates = [x.value for x in collision_data]
        
        plt.subplot(2, 2, 4)
        plt.plot(steps, rates)
        plt.title("Collision Rate")
        plt.xlabel("Timesteps")
        plt.ylabel("Collision Rate")
    
    plt.tight_layout()
    plt.savefig("./logs/training_performance.png")
    print("Saved training performance plot to logs/training_performance.png")
    print(f"Generated from: {os.path.basename(latest_file)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drone Navigation Training and Testing")
    parser.add_argument("--mode", choices=["train", "test", "plot"], required=True, help="Mode to run: train, test, or plot")
    parser.add_argument("--gui", action="store_true", help="Enable GUI during training")
    parser.add_argument("--camera", choices=["static", "follow"], default="follow", help="Camera mode during testing")
    args = parser.parse_args()

    if args.mode == "train":
        train(use_gui=args.gui)
        plot_training_results()  # Generate plots after training
    elif args.mode == "test":
        test_model(camera_mode=args.camera)
    elif args.mode == "plot":
        plot_training_results()