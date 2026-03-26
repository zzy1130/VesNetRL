#!/usr/bin/env python3
"""Quick sim test: run author's pre-trained model on simulated vessels.
No visualization, just print success rate + action distribution."""
import numpy as np
import torch
import torch.nn.functional as F
from Env import Env_multi_sim_img_test
from model import VesNet_RL
from collections import Counter

n_episodes = 50
max_step = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

def create_configs_rand(num):
    configs = []
    r_min, r_max = 45, 65
    for i in range(num):
        offset = np.random.rand() * np.pi / 2
        size_3d = [750, 700, 450]
        r = np.random.randint(r_min + (r_max - r_min) * i / num,
                              r_min + (r_max - r_min) * (i + 1) / num)
        c_x = 350
        c_y = np.random.randint(r, 450 - r)
        config = ([c_x, c_y], r, size_3d, offset)
        configs.append(config)
    return configs

print("Creating sim environment...")
configs = create_configs_rand(1)
env = Env_multi_sim_img_test(configs=configs, num_channels=4)

print("Loading author's pre-trained weights...")
model = VesNet_RL(env.num_channels, 5, env.num_actions).to(device)
model.load_state_dict(torch.load('VesNet_RL_ckpt/trained_model/checkpoint.pth', map_location=device))
# NOTE: do NOT call model.eval() -- author's code keeps train mode
# BatchNorm with batch_size=1 in train mode acts as instance norm,
# which is how the model was trained

ACTION_NAMES = ['Stay', '+Along', '-Along', '+Perp', '-Perp', '+Rot', '-Rot']
all_actions = []
num_success = 0

print(f"\nRunning {n_episodes} episodes (max {max_step} steps each)...\n")
for ep in range(1, n_episodes + 1):
    state = env.reset(randomVessel=True)
    cx = torch.zeros(1, 256).float().to(device)
    hx = torch.zeros(1, 256).float().to(device)
    actions = []
    done = False

    for step in range(max_step):
        with torch.no_grad():
            value, logit, (hx, cx) = model((state, (hx, cx)))
        prob = F.softmax(logit, dim=-1)
        action = prob.multinomial(num_samples=1).cpu().detach().numpy()
        a = int(action)
        actions.append(a)
        state, done = env.step(a)
        if done:
            num_success += 1
            break

    all_actions.extend(actions)
    status = "OK" if done else "--"
    print(f"  Ep {ep:3d}: {status}  steps={len(actions):2d}  actions={actions}")

# Summary
print(f"\n{'='*60}")
print(f"Success: {num_success}/{n_episodes} ({num_success/n_episodes*100:.1f}%)")
print(f"\nAction distribution:")
counts = Counter(all_actions)
for i in range(7):
    bar = '#' * (counts.get(i, 0) // 5)
    print(f"  {i} ({ACTION_NAMES[i]:>6s}): {counts.get(i, 0):4d}  {bar}")
print(f"  Total actions: {len(all_actions)}")
