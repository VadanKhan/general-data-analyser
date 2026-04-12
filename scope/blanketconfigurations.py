import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ==========================================
# --- USER CONFIGURATION SECTION ---
# ==========================================

DATA_FILENAME = "trimmed_TBR_optimisation_results.csv"
DATA_FOLDER = Path("data/")
FIGURES_FOLDER = Path("figures/")
OUTPUT_NAME = "breeder_blanket_2d_themed.png"

# Visual Settings
LAYER_COLS = ['B1', 'C1', 'B2', 'C2', 'B3', 'C3', 'B4', 'C4']
ANNOTATION_THRESHOLD = 2.0  # cm. Below this, text moves off the bar

# Theming: Breeders (B) = Reds, Coolants (C) = Blues
REDS = ['#67000d', '#a50f15', '#ef3b2c', '#fc9272']
BLUES = ['#08306b', '#2171b5', '#6baed6', '#c6dbef']

DPI = 500
TITLE = "Optimised Thickness Configurations & TBR"

# ==========================================
# --- PLOTTING LOGIC ---
# ==========================================


def main():
    # 1. Load Data
    data_path = DATA_FOLDER / DATA_FILENAME
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"\n[!] Error: Could not find '{data_path}'.")
        return

    run_names = df['Runname'].tolist()
    tbr_values = df['TBR'].tolist()
    tbr_unc = df['Unc'].tolist()

    # Generate interleaved color map
    colors = []
    for col in LAYER_COLS:
        idx = int(col[1]) - 1
        colors.append(REDS[idx] if 'B' in col else BLUES[idx])

    # 2. Setup Figure (Restored width to 12 since text is stacked)
    fig, ax = plt.subplots(figsize=(12, 6))

    y_pos = np.arange(len(run_names))
    left_positions = np.zeros(len(run_names))
    bar_height = 0.6

    # 3. Draw Stacked Bars
    for i, col in enumerate(LAYER_COLS):
        widths = df[col].values

        # Plot horizontal bar segment
        ax.barh(y_pos, widths, height=bar_height, left=left_positions,
                color=colors[i], edgecolor='black', label=col)

        # Add Annotations
        for y_idx, (w, l) in enumerate(zip(widths, left_positions)):
            x_center = l + w / 2
            y_center = y_idx

            if w > ANNOTATION_THRESHOLD:
                # Text inside the bar (Make text white for the darkest colors)
                text_col = 'white' if i < 2 else 'black'
                ax.text(x_center, y_center, f'{w:.1f}',
                        ha='center', va='center', color=text_col, fontsize=10)
            elif w > 0:
                ANNOTATION_HEIGHT = 0.15
                # Thin layer: Draw an anchor line pointing up, text above the bar
                ax.plot([x_center, x_center], [y_center + bar_height / 2, y_center + bar_height / 2 + ANNOTATION_HEIGHT],
                        color='black', linewidth=1)
                ax.text(x_center, y_center + bar_height / 2 + 0.25, f'{w:.1f}',
                        ha='center', va='bottom', color='black', fontsize=9, style='italic')

        left_positions += widths

    # 4. Format Layout
    ax.set_yticks(y_pos)
    ax.set_yticklabels(run_names, fontsize=11)
    ax.set_xlabel('Radial Thickness (cm)', fontsize=12)
    ax.set_title(TITLE, fontsize=15, pad=15)

    ax.invert_yaxis()  # Top-down reading (matches dataframe order)

    # 5. Right-Justified Bold TBR
    max_total_width = left_positions.max()

    # REVERTED: Scaled this back down to 1.25 to remove the dead space on the right
    ax.set_xlim(0, max_total_width * 1.25)

    for y_idx, (tbr, unc) in enumerate(zip(tbr_values, tbr_unc)):
        # NEW: Added '\n' before the ± symbol to stack the text on two lines
        ax.text(max_total_width + (max_total_width * 0.02), y_idx, f'TBR: {tbr:.4f}\n± {unc:.4f}',
                ha='left', va='center', color='darkred', fontsize=12, fontweight='bold')

    # 6. Legends and Clean up
    legend_patches = [
        mpatches.Patch(color=colors[i], label=LAYER_COLS[i]) for i in range(len(LAYER_COLS))
    ]
    ax.legend(
        handles=legend_patches,
        bbox_to_anchor=(
            1.02,
            1),
        loc='upper left',
        title="Blanket Layers")

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # 7. Save Output
    FIGURES_FOLDER.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_FOLDER / OUTPUT_NAME

    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches='tight')
    print(f"\n[+] Success! 2D themed plot saved to {out_path}")


if __name__ == "__main__":
    main()
