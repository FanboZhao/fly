import os
os.environ['MUJOCO_GL']='egl'

from flybody.fly_envs import flight_imitation
from flybody.agents.agent_dmpo import DMPO
from flybody.agents.network_factory_vis import make_vis_network_factory_two_level_controller
from flybody.agents.network_factory import make_network_factory_dmpo
from flybody.agents.utils_tf import restore_dmpo_networks_from_checkpoint
from acme import specs, wrappers
import numpy as np
import tensorflow as tf

# from dm_control import viewaer
from tqdm import tqdm
import mediapy
import matplotlib.pyplot as plt
from direction_tracking_env import direction_tracking

# The dictionary of stored checkpoints
ckpt_dir = "checkpoints/plume_tracking"
os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs("video", exist_ok=True)

def wrap_env(env):
    """Wrap task environment with Acme wrappers."""
    return wrappers.CanonicalSpecWrapper(
        wrappers.SinglePrecisionWrapper(env),
        clip=True)

wpg_pattern_path = 'flybody-data/datasets_flight-imitation/wing_pattern_fmech.npy'
# high_level_ckpt_path = 'flybody-data/flight-controller-reuse-checkpoints/checkpoints/high-level-controllers/trench-task/ckpt-48'
low_level_ckpt_path = 'flybody-data/flight-controller-reuse-checkpoints/checkpoints/low-level-controller/ckpt-11'

goal_position = np.array([1, -3, 1])

env = direction_tracking(
        wpg_pattern_path=wpg_pattern_path,
        bumps_or_trench='bumps',
        joint_filter=0.0002,
        goal_position=goal_position
    )

env = wrap_env(env)

# Preview the environment
# viewer.launch(env)

environment_spec = specs.make_environment_spec(env)

ll_env = wrap_env(flight_imitation(
    future_steps=5,
    joint_filter=0.0002))

ll_environment_spec = specs.make_environment_spec(ll_env)

ll_network_factory = make_network_factory_dmpo()


# Create networks for the vision flight task from their network factory.
network_factory = make_vis_network_factory_two_level_controller(
    ll_network_ckpt_path=low_level_ckpt_path,
    ll_network_factory=ll_network_factory,
    ll_environment_spec=ll_environment_spec,
    hl_network_layer_sizes=(256, 256, 128),
    steering_command_dim=(5 + 1) * (3 + 4),
    task_input_dim=3,
    vis_output_dim=8,
    critic_layer_sizes=(512, 512, 256),
)

networks = network_factory(env.action_spec())

# networks = restore_dmpo_networks_from_checkpoint(
#     ckpt_path=high_level_ckpt_path,
#     network_factory=network_factory,
#     environment_spec=environment_spec)

agent = DMPO(environment_spec=environment_spec,
            policy_network=networks['policy'],
            critic_network=networks['critic'],
            observation_network=networks['observation'],
            )

# agent = DMPO(environment_spec=environment_spec,
#              policy_network=networks.policy_network,
#              critic_network=networks.critic_network,
#              observation_network=networks.observation_network,
#             )

checkpoint = tf.train.Checkpoint(
    policy=networks['policy'],
    critic=networks['critic'],
    observation=networks['observation']  # 保存 observation_network
)

# checkpoint = tf.train.Checkpoint(
#     policy_network=networks.policy_network,
#     critic_network=networks.critic_network,
#     observation_network=networks.observation_network,
# )
manager = tf.train.CheckpointManager(checkpoint, ckpt_dir, max_to_keep=3)
checkpoint.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print(f"✅ Succussfully restore from {manager.latest_checkpoint} ")
else:
    print("⚠️ Can't find latest checkpoint")

timestep = env.reset()
agent.observe_first(timestep)

global_step = 0
reward = 0
reward_history = []
step_history = []

"""Begin training"""
  
for i in range(5):
    timestep = env.reset()
    
    step = 0
    max_steps = 10000

    while step < max_steps:
        action = agent.select_action(timestep.observation)
        next_timestep = env.step(action)
        agent.observe(action, next_timestep)
        agent.update()
        reward = next_timestep.reward
        reward_history.append(reward)
        step_history.append(global_step)
        if next_timestep.last():
            timestep = env.reset()
        else:
            timestep = next_timestep
        step += 1
        global_step += 1

    print(f"Step {(i + 1) * 10000} completed")
    print(f"reward : {reward}")
    save_path = manager.save()
    print(f"✅ Checkpoint saved at: {save_path}")

env.close()

plt.figure(figsize=(10, 5))
plt.plot(step_history, reward_history, label="Reward per step", alpha=0.6)

window = 100
if len(reward_history) > window:
    moving_avg = np.convolve(reward_history, np.ones(window)/window, mode='valid')
    plt.plot(step_history[window-1:], moving_avg, label=f"{window}-step moving avg", color="red")

plt.xlabel("Step")
plt.ylabel("Reward")
plt.title("Training Reward over Steps")
plt.legend()
plt.grid(True)
plt.savefig("video/reward_curve.png", dpi=300)

"""Save as a video"""
frames = []
timestep = env.reset() 
for _ in tqdm(range(500)):
    action = agent.select_action(timestep.observation)
    timestep = env.step(action)
    cam_id = env.physics.model.name2id('tracking_cam', 'camera')
    frames.append(env.physics.render(camera_id=cam_id, width=640, height=480))

mediapy.write_video('video/direction_tracking.mp4', frames, fps=30)
env.close()
