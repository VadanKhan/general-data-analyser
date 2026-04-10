from matplotlib.patches import Patch
import matplotlib.pyplot as plt

from pathlib import Path
DATA_FOLDER = Path("data/")
FIGURES_FOLDER = Path("figures/")
NAME = "thermal_efficiency_comparison"

# 1. The Data
reactors = ['Magnox', 'BWR', 'PWR', 'EPR', 'FBR (Sodium)', 'AGR', 'MSR/MSFR']
efficiencies = [31, 33, 34, 36, 40, 41, 46]  # % Thermal Efficiency
temperatures = [400, 285, 315, 327, 550, 650, 750]  # Core Outlet Temp in °C

# 2. Color Coding for visual storytelling
# Grey = Legacy, Blue = Water (Current), Orange/Gold = Advanced/Future
colors = ['#95a5a6', '#3498db', '#3498db', '#2980b9', '#f39c12', '#7f8c8d', '#e67e22']

# 3. Setup the Figure
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)  # High DPI for crisp presentation slides
bars = ax.barh(reactors, efficiencies, color=colors, edgecolor='black', linewidth=0.8)

# 4. Customizing the Aesthetics
ax.set_xlabel('Thermal Efficiency (%)', fontsize=12, fontweight='bold', color='#333333')
ax.set_title('Nuclear Reactor Thermal Efficiency',
             fontsize=16, fontweight='bold', pad=20, color='#2c3e50')

# Remove top and right borders for a cleaner look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#bdc3c7')
ax.spines['bottom'].set_color('#bdc3c7')

# Add a subtle vertical grid
ax.xaxis.grid(True, linestyle='--', alpha=0.6, color='#bdc3c7')
ax.set_axisbelow(True)  # Puts grid behind bars

# 5. Add Data Labels (Efficiency % and Operating Temp)
for bar, temp in zip(bars, temperatures):
    width = bar.get_width()
    # Add the percentage inside the bar
    ax.text(width - 0.5, bar.get_y() + bar.get_height() / 2, f'{width}%',
            ha='right', va='center', color='white', fontweight='bold', fontsize=11)
    # Add the operating temperature outside the bar
    ax.text(width + 0.5, bar.get_y() + bar.get_height() / 2, f'~{temp}°C',
            ha='left', va='center', color='#555555', fontsize=10, fontstyle='italic')

# 6. Add a custom legend to explain the colors
legend_elements = [
    Patch(facecolor='#95a5a6', edgecolor='black', label='Legacy UK Fleet (Gas)'),
    Patch(facecolor='#3498db', edgecolor='black', label='Current Standard (Water)'),
    Patch(facecolor='#e67e22', edgecolor='black', label='Generation IV (Salt/Metal)')
]
ax.legend(handles=legend_elements, loc='best', fontsize=10, frameon=True, borderpad=1)

plt.tight_layout()
plt.savefig(FIGURES_FOLDER / f'{NAME}.png', bbox_inches='tight', dpi=500)

# plt.show()
