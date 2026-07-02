import argparse
import time
import numpy as np
import cv2
from env_config import make_env

# Import all controllers
from random_controller import RandomController
from rule_based_controller import RuleBasedController
from ppo_controller_frame import PPOController
from dqn_controller import DQNController


def record_controller(controller, label, n_episodes, delay=0.10):
    """
    Runs a single controller for N episodes and saves the entire run 
    as a single MP4 video file.
    """
    print(f"\n[{label}] Starting recording for {n_episodes} episodes...")
    
    # Create a fresh environment for this controller
    env = make_env(render_mode="rgb_array")
    
    video_writer = None
    video_name = f"{label}_evaluation.mp4"

    for ep in range(n_episodes):
        print(f"  -> Recording Episode {ep + 1}/{n_episodes}...")
        
        # Use a consistent seed for a fair comparison across all videos
        obs, info = env.reset(seed=ep + 42)
        controller.reset(seed=ep + 42)
        
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            # 1. Get action from controller
            action, _ = controller.act(obs)
            
            # 2. Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 3. Capture the frame
            frame = env.render()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Add text overlay
            cv2.putText(frame_bgr, f"{label} - Episode {ep+1}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
            if terminated or truncated:
                status = "CRASHED" if info.get("crashed", False) else "FINISHED"
                cv2.putText(frame_bgr, status, (20, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

            # 4. Initialize video writer on the very first frame
            if video_writer is None:
                fps = int(1.0 / delay)
                h, w = frame_bgr.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(video_name, fourcc, fps, (w, h))
                
            # Write frame to video
            video_writer.write(frame_bgr)

    # Clean up and save the video
    if video_writer is not None:
        video_writer.release()
    env.close()
    
    print(f"[{label}] Video successfully saved as '{video_name}'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppo_model", type=str, default="models/ppo_final")
    parser.add_argument("--dqn_model", type=str, default="models/dqn_final")
    parser.add_argument("--controller", type=str, default="all",
                        choices=["all", "random", "rule_based", "ppo", "dqn"],
                        help="Which controller to record")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--delay", type=float, default=0.10, help="Simulated frame delay (FPS)")
    args = parser.parse_args()

    print("Loading controllers...")

    if args.controller == "all":
        controllers = [
            ("Random", RandomController()),
            ("Rule-Based", RuleBasedController()),
            ("PPO", PPOController(args.ppo_model)),
            ("DQN", DQNController(args.dqn_model))
        ]
    elif args.controller == "random":
        controllers = [("Random", RandomController())]
    elif args.controller == "rule_based":
        controllers = [("Rule-Based", RuleBasedController())]
    elif args.controller == "ppo":
        controllers = [("PPO", PPOController(args.ppo_model))]
    else:
        controllers = [("DQN", DQNController(args.dqn_model))]
    
    # Process them one by one to avoid Pygame crashes
    for label, ctrl in controllers:
        record_controller(ctrl, label, args.episodes, args.delay)
        
    print("\nAll videos have been generated successfully!")


if __name__ == "__main__":
    main()