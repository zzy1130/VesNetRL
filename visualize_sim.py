#!/usr/bin/env python3
"""
VesNet-RL Simulation Visualizer
Shows the agent navigating the US probe to find the best vessel cross-section.

Left:  Top-down view of the vessel with probe position & trail
Right: The 2D ultrasound slice the agent currently sees
"""
import numpy as np
import torch
import torch.nn.functional as F
import math
import cv2
import matplotlib
matplotlib.use('macosx')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from collections import deque

from Env import Env_multi_sim_img_test
from model import VesNet_RL

# ── Settings ──────────────────────────────────────────────────────────
MODEL_PATH = 'VesNet_RL_ckpt/trained_model/checkpoint.pth'
N_EPISODES = 5
MAX_STEPS = 50
PAUSE_SEC = 0.15        # seconds between steps (controls playback speed)
NUM_CHANNELS = 4

ACTION_NAMES = ['Stay', '+Along', '-Along', '+Perp', '-Perp', '+Rot', '-Rot']
ACTION_ARROWS = ['●', '→', '←', '↑', '↓', '↻', '↺']

# ── Setup ─────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

print("Creating vessel environment (this takes ~30s on CPU)...")
configs = create_configs_rand(1)
env = Env_multi_sim_img_test(configs=configs, num_channels=NUM_CHANNELS)

print("Loading model...")
model = VesNet_RL(env.num_channels, 5, env.num_actions).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
# Do NOT call model.eval() -- BatchNorm with batch_size=1 must stay in train mode

# ── Visualization ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5),
                         gridspec_kw={'width_ratios': [1.2, 1, 0.6]})
fig.suptitle('VesNet-RL: Ultrasound Probe Navigation Simulator', fontsize=14, fontweight='bold')
plt.subplots_adjust(wspace=0.3, left=0.05, right=0.95, top=0.88, bottom=0.08)

ax_top = axes[0]   # top-down view
ax_us = axes[1]    # US slice
ax_info = axes[2]  # info panel

def draw_probe(ax, pos, probe_width, color='lime', linewidth=2):
    """Draw the probe line on the top-down view."""
    x, y, theta = pos
    dx = probe_width / 2 * math.cos(theta)
    dy = probe_width / 2 * math.sin(theta)
    ax.plot([y - dy, y + dy], [x - dx, x + dx], color=color, linewidth=linewidth, solid_capstyle='round')

def run_visualization():
    for ep in range(N_EPISODES):
        state = env.reset(randomVessel=True)
        vessel = env.vessels[env.cur_env]

        # Get a z-slice of the 3D vessel for the top-down background
        mid_z = vessel.size_3d[2] // 2
        topdown_bg = vessel.img[:, :, mid_z].copy()

        # Overlay the valid mask
        mask_overlay = np.stack([topdown_bg, topdown_bg, topdown_bg], axis=-1)
        mask_contour = (vessel.mask * 0.15)
        mask_overlay[:, :, 1] = np.clip(mask_overlay[:, :, 1] + mask_contour, 0, 1)

        cx = torch.zeros(1, 256).float().to(device)
        hx = torch.zeros(1, 256).float().to(device)

        trail_x, trail_y = [], []
        done = False

        for step in range(MAX_STEPS):
            pos = env.pos.copy()
            trail_x.append(pos[1])  # y -> horizontal in plot
            trail_y.append(pos[0])  # x -> vertical in plot

            # ── Left: Top-down view ──
            ax_top.clear()
            ax_top.imshow(mask_overlay, origin='upper', extent=[0, vessel.size_3d[1], vessel.size_3d[0], 0])
            ax_top.set_title('Top-Down View (X-Y Plane)', fontsize=11)
            ax_top.set_xlabel('Y (pixels)')
            ax_top.set_ylabel('X (pixels)')

            # Draw vessel center line
            ax_top.axhline(y=vessel.c[0], color='cyan', linewidth=0.8, linestyle='--', alpha=0.5, label='Vessel center')

            # Draw trail
            if len(trail_x) > 1:
                ax_top.plot(trail_x, trail_y, 'o-', color='yellow', markersize=2, linewidth=1, alpha=0.6)

            # Draw current probe
            draw_probe(ax_top, pos, vessel.probe_width, color='lime', linewidth=2.5)
            ax_top.plot(pos[1], pos[0], 'o', color='red', markersize=6, zorder=10)

            # Legend
            legend_elements = [
                Line2D([0], [0], color='cyan', linestyle='--', label='Vessel center'),
                Line2D([0], [0], color='lime', linewidth=2.5, label='US probe'),
                Line2D([0], [0], color='yellow', marker='o', markersize=3, label='Trail'),
                Line2D([0], [0], color='red', marker='o', linestyle='', markersize=5, label='Probe center'),
            ]
            ax_top.legend(handles=legend_elements, loc='lower right', fontsize=7, framealpha=0.8)

            # ── Middle: US slice image ──
            ax_us.clear()
            # Current image from environment
            us_image = env.image if hasattr(env, 'image') else np.zeros((256, 256))
            ax_us.imshow(us_image, cmap='gray', vmin=0, vmax=1)
            ax_us.set_title('Ultrasound Slice (Agent View)', fontsize=11)
            ax_us.set_xlabel('Width')
            ax_us.set_ylabel('Depth')

            # Compute vessel area
            vessel_area = len(np.where(us_image > 0.9)[0])

            # ── Right: Info panel ──
            ax_info.clear()
            ax_info.axis('off')
            ax_info.set_xlim(0, 1)
            ax_info.set_ylim(0, 1)

            info_lines = [
                f"Episode: {ep + 1} / {N_EPISODES}",
                f"Step: {step} / {MAX_STEPS}",
                f"",
                f"Probe Position:",
                f"  x = {pos[0]}",
                f"  y = {pos[1]}",
                f"  θ = {pos[2]:.2f} rad",
                f"",
                f"Vessel (r={vessel.r}):",
                f"  center = {vessel.c}",
                f"  area = {vessel_area} px",
                f"",
                f"Status: {'FOUND!' if done else 'Searching...'}",
            ]

            if step > 0:
                info_lines.insert(7, f"  Action: {ACTION_ARROWS[last_action]} {ACTION_NAMES[last_action]}")

            for i, line in enumerate(info_lines):
                color = 'limegreen' if 'FOUND' in line else 'white'
                fontweight = 'bold' if 'FOUND' in line or 'Episode' in line else 'normal'
                fontsize = 11 if 'FOUND' in line else 9
                ax_info.text(0.05, 0.95 - i * 0.065, line, fontsize=fontsize,
                           fontfamily='monospace', color=color, fontweight=fontweight,
                           verticalalignment='top', transform=ax_info.transAxes)

            ax_info.set_facecolor('#1a1a2e')
            fig.patch.set_facecolor('#0f0f23')
            ax_top.set_facecolor('#0a0a1a')

            plt.draw()
            plt.pause(PAUSE_SEC)

            if done:
                # Flash success
                for _ in range(3):
                    fig.patch.set_facecolor('#004400')
                    plt.draw()
                    plt.pause(0.2)
                    fig.patch.set_facecolor('#0f0f23')
                    plt.draw()
                    plt.pause(0.2)
                break

            # ── Agent action ──
            with torch.no_grad():
                value, logit, (hx, cx) = model((state, (hx, cx)))
            prob = F.softmax(logit, dim=-1)
            action = prob.multinomial(num_samples=1).cpu().detach().numpy()
            last_action = int(action)
            state, done = env.step(last_action)

        print(f"Episode {ep+1}: {'SUCCESS' if done else 'TIMEOUT'} in {step+1} steps")

    print("\nAll episodes done. Close the window to exit.")
    plt.show()

if __name__ == '__main__':
    run_visualization()
