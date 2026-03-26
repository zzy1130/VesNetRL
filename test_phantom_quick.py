#!/usr/bin/env python3
"""Quick phantom test: headless, print success rate."""
import numpy as np
import torch
import torch.nn.functional as F
from Env import Env_multi_re_img_a2c_test
from model import VesNet_RL
from collections import Counter

n_episodes = 20
max_step = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading environment (vessel_1 + UNet, points_interval=20)...")
n1_img = [['./3d_models/vessel_1.nii', np.pi/2]]
env = Env_multi_re_img_a2c_test(n1_img=n1_img, num_channels=4, points_interval=20)
print("Environment ready.")

model = VesNet_RL(env.num_channels, 5, env.num_actions).to(device)
model.load_state_dict(torch.load('VesNet_RL_ckpt/trained_model/checkpoint.pth', map_location=device))
# Do NOT call model.eval()

ACTION_NAMES = ['Stay', '+Along', '-Along', '+Perp', '-Perp', '+Rot', '-Rot']
all_actions = []
num_success = 0

print(f"\nRunning {n_episodes} episodes...\n")
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
        action = int(prob.multinomial(num_samples=1).cpu().detach().numpy().item())
        actions.append(action)
        state, done = env.step(action)
        if done:
            num_success += 1
            break
    all_actions.extend(actions)
    status = "OK" if done else "--"
    print(f"  Ep {ep:3d}: {status}  steps={len(actions):2d}")

print(f"\nSuccess: {num_success}/{n_episodes} ({num_success/n_episodes*100:.1f}%)")
counts = Counter(all_actions)
print("Action distribution:")
for i in range(7):
    print(f"  {ACTION_NAMES[i]:>6s}: {counts.get(i, 0):4d}")
