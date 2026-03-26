#!/usr/bin/env python3
"""
VesNet-RL Phantom Visualizer (Real Ultrasound Texture)

Uses the ORIGINAL Env_multi_re_img_a2c_test environment directly,
with visualization hooks to display raw US + UNet segmentation.
"""
import numpy as np
import torch
import torch.nn.functional as F
import math
import cv2
import matplotlib
matplotlib.use('macosx')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import deque
import torchvision.transforms as transforms

from Env import Env_multi_re_img_a2c_test
from model import VesNet_RL

# ── Settings ──────────────────────────────────────────────────────────
MODEL_PATH = 'VesNet_RL_ckpt/trained_model/checkpoint.pth'
N_EPISODES = 10
MAX_STEPS = 50
PAUSE_SEC = 0.25

ACTION_NAMES = ['Stay', '+Along', '-Along', '+Perp', '-Perp', '+Rot', '-Rot']
ACTION_ARROWS = ['●', '→', '←', '↑', '↓', '↻', '↺']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load environment (ORIGINAL class, points_interval=20) ────────────
# Only use vessel_1 to reduce init time; set points_interval=20 for accuracy
print("Loading NIfTI vessel + UNet + computing centerline...")
print("(points_interval=20, accurate but takes ~2-3 min on CPU)")
n1_img = [['./3d_models/vessel_1.nii', np.pi/2]]
env = Env_multi_re_img_a2c_test(n1_img=n1_img, num_channels=4, points_interval=20)
print("Environment ready.")

# ── Load RL model (author's pre-trained weights) ─────────────────────
print(f"Loading model from {MODEL_PATH}...")
model = VesNet_RL(env.num_channels, 5, env.num_actions).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
# Do NOT call model.eval() -- BatchNorm with batch_size=1 must stay in train mode
print("Model loaded.\n")

# ── Image transform (for displaying raw US) ──────────────────────────
transform_image = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])

# ── Visualization setup ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6),
                         gridspec_kw={'width_ratios': [1, 1, 1]})
fig.suptitle('VesNet-RL: Real Ultrasound Phantom Navigation (Author Weights)',
             fontsize=14, fontweight='bold', color='white')
plt.subplots_adjust(wspace=0.25, left=0.04, right=0.96, top=0.88, bottom=0.08)

def draw_probe(ax, pos, probe_width, color='lime', linewidth=2):
    x, y, theta = pos[0], pos[1], pos[2]
    dx = probe_width / 2 * math.cos(theta)
    dy = probe_width / 2 * math.sin(theta)
    ax.plot([y - dy, y + dy], [x - dx, x + dx],
            color=color, linewidth=linewidth, solid_capstyle='round')

# ── Run ───────────────────────────────────────────────────────────────
num_success = 0

for ep in range(N_EPISODES):
    print(f"── Episode {ep+1}/{N_EPISODES} ──")

    state = env.reset(randomVessel=True)
    vessel = env.vessels[env.cur_env]

    cx_h = torch.zeros(1, 256).float().to(device)
    hx_h = torch.zeros(1, 256).float().to(device)

    trail_x, trail_y = [env.pos[1]], [env.pos[0]]
    done = False
    last_action = 0

    for step in range(MAX_STEPS):
        pos = env.pos

        # ── Get raw US image for display (env.image is the raw slice) ──
        raw_img = env.image  # raw US from get_slicer()
        # UNet segmentation is stored after step/reset
        seg_img = env.pred_th if hasattr(env, 'pred_th') else np.zeros((256, 256))

        # ── Draw ──
        # Left: Raw ultrasound
        axes[0].clear()
        if raw_img is not None and raw_img.size > 0:
            axes[0].imshow(raw_img, cmap='gray', vmin=0, vmax=1)
        else:
            axes[0].imshow(np.zeros((256, 256)), cmap='gray')
        axes[0].set_title('Raw Ultrasound Image', fontsize=12, color='white')
        axes[0].set_xlabel('Width (px)', color='#aaa')
        axes[0].set_ylabel('Depth (px)', color='#aaa')
        axes[0].tick_params(colors='#888')

        # Middle: UNet segmentation overlay
        axes[1].clear()
        if raw_img is not None and raw_img.size > 0 and seg_img is not None:
            base = np.clip(raw_img, 0, 1)
            overlay = np.stack([base, base, base], axis=-1)
            seg_mask = seg_img > 0.5
            overlay[seg_mask, 0] = overlay[seg_mask, 0] * 0.3
            overlay[seg_mask, 1] = np.clip(overlay[seg_mask, 1] * 0.3 + 0.7, 0, 1)
            overlay[seg_mask, 2] = overlay[seg_mask, 2] * 0.3
            axes[1].imshow(overlay)
        else:
            axes[1].imshow(np.zeros((256, 256, 3)))
        axes[1].set_title('UNet Segmentation Overlay', fontsize=12, color='white')
        axes[1].set_xlabel('Width (px)', color='#aaa')
        axes[1].tick_params(colors='#888')

        # Vessel area
        vessel_area = len(np.where(seg_img > 0.9)[0]) if seg_img is not None else 0
        info_text = f"Ep {ep+1}/{N_EPISODES}  Step {step}/{MAX_STEPS}  Area: {vessel_area}px"
        if step > 0:
            info_text += f"\nAction: {ACTION_ARROWS[last_action]} {ACTION_NAMES[last_action]}"
        if done:
            info_text += "\n** VESSEL FOUND! **"
        axes[1].text(5, 15, info_text, fontsize=9, color='yellow',
                    fontfamily='monospace', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

        # Right: Top-down map
        axes[2].clear()
        axes[2].imshow(vessel.center_line, cmap='hot', origin='upper',
                      extent=[0, vessel.img.shape[1], vessel.img.shape[0], 0],
                      alpha=0.8)
        if len(trail_x) > 1:
            axes[2].plot(trail_x, trail_y, 'o-', color='cyan', markersize=2,
                        linewidth=1, alpha=0.7)
        draw_probe(axes[2], pos, vessel.probe_width, color='lime', linewidth=2)
        axes[2].plot(pos[1], pos[0], 'o', color='red', markersize=6, zorder=10)
        if hasattr(vessel, 'goal_pos'):
            gp = vessel.goal_pos
            axes[2].plot(gp[1], gp[0], '*', color='gold', markersize=14, zorder=10)
        axes[2].set_title('Vessel Map + Probe Trail', fontsize=12, color='white')
        axes[2].set_xlabel('Y (px)', color='#aaa')
        axes[2].set_ylabel('X (px)', color='#aaa')
        axes[2].tick_params(colors='#888')
        legend_elements = [
            Line2D([0], [0], color='lime', linewidth=2, label='Probe'),
            Line2D([0], [0], color='cyan', marker='o', markersize=3, label='Trail'),
            Line2D([0], [0], color='gold', marker='*', linestyle='', markersize=8, label='Goal'),
        ]
        axes[2].legend(handles=legend_elements, loc='lower right', fontsize=7,
                      framealpha=0.7, facecolor='black', labelcolor='white')

        fig.patch.set_facecolor('#0f0f23')
        for ax in axes:
            ax.set_facecolor('#0a0a1a')
            for spine in ax.spines.values():
                spine.set_color('#333')

        plt.draw()
        plt.pause(PAUSE_SEC)

        if done:
            num_success += 1
            for _ in range(4):
                fig.patch.set_facecolor('#003300')
                plt.draw()
                plt.pause(0.15)
                fig.patch.set_facecolor('#0f0f23')
                plt.draw()
                plt.pause(0.15)
            break

        # ── Agent step (EXACTLY like original test script) ──
        with torch.no_grad():
            value, logit, (hx_h, cx_h) = model((state, (hx_h, cx_h)))
        prob = F.softmax(logit, dim=-1)
        action = prob.multinomial(num_samples=1).cpu().detach().numpy()
        last_action = int(action)

        state, done = env.step(last_action)

        trail_x.append(env.pos[1])
        trail_y.append(env.pos[0])

    status = "SUCCESS" if done else "TIMEOUT"
    print(f"  {status} in {step+1} steps  (pos: x={env.pos[0]:.0f} y={env.pos[1]:.0f} θ={env.pos[2]:.2f})")

print(f"\nDone. Success: {num_success}/{N_EPISODES} ({num_success/N_EPISODES*100:.0f}%)")
print("Close the window to exit.")
plt.show()
