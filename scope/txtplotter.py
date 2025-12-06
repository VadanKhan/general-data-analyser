import numpy as np
import matplotlib.pyplot as plt

# NOTE!!!!
# DON'T USE PLOTLY FOR .py scripts, PLOTLY NEEDS .ipynb INTERACTIVE ENVIRONMENTS.

# import plotly.io as pio
# import plotly.graph_objects as go


from pathlib import Path
DATA_FOLDER = Path("data/")
FIGURES_FOLDER = Path("figures/")
NAME = "lithium_cs"
TITLE = None

# pio.renderers.default = "vscode"

try:
    data1 = np.genfromtxt(DATA_FOLDER / "Data Li-6", dtype='float', delimiter=',', skip_header=0)
    x_data1 = data1[:, 1]
    y_data1 = data1[:, 2]

    data2 = np.genfromtxt(DATA_FOLDER / "Data Li-7", dtype='float', delimiter=',', skip_header=0)
    x_data2 = data2[:, 1]
    y_data2 = data2[:, 2]

except (IOError, FileNotFoundError) as e:
    print(f"Error: Could not find or read data files in '{DATA_FOLDER}'.")
    print(f"Details: {e}")
    # You might want to exit or use dummy data here
    # For this example, we'll stop if data isn't found.
    exit()
except IndexError:
    print("Error: Data files seem to be empty or have an incorrect format.")
    exit()

plt.plot(x_data1 / 1E6, y_data1, '-', color='blue', label='$^6$Li')
plt.plot(x_data2 / 1E6, y_data2, '-', color='red', label='$^7$Li')

# plt.xlim(0, 15)
plt.ylim(1E-4, 1E1)
plt.grid()
plt.yscale('log')
plt.legend()

plt.title(TITLE)
plt.xlabel('Incident Neutron Energy (MeV)')
plt.ylabel('$\\sigma$ (barns)')

plt.savefig(FIGURES_FOLDER / f'{NAME}.png', dpi=500)

plt.show()


# # --- 1. Initialize a Plotly Figure ---
# fig = go.Figure()

# # --- 2. Add Traces (instead of plt.plot) ---

# # Add the ⁶Li trace
# fig.add_trace(go.Scatter(
#     x=x_data1 / 1E6,
#     y=y_data1,
#     mode='lines',  # Equivalent to the '-' in 'b-'
#     name='⁶Li'       # Sets the legend label (handles unicode)
#     # No color is specified, Plotly assigns the first default color
# ))

# # Add the ⁷Li trace
# fig.add_trace(go.Scatter(
#     x=x_data2 / 1E6,
#     y=y_data2,
#     mode='lines',
#     name='⁷Li'
#     # No color is specified, Plotly assigns the second default color
# ))

# # --- 3. Update Layout (replaces plt.title, plt.xlabel, etc.) ---
# fig.update_layout(
#     title='Tritium Production Cross Sections of Lithium Isotopes',
#     xaxis_title='Incident Neutron Energy (MeV)',
#     yaxis_title='σ (barns)',  # Plotly handles Greek letters like σ
#     yaxis_type='log',         # Replaces plt.yscale('log')
#     yaxis_range=[-4, 1],      # Replaces plt.ylim(1E-4, 1E1)
#                               # For log scale, range is in powers of 10:
#                               # log10(1E-4) = -4
#                               # log10(1E1)  =  1
#     legend_title_text='Isotopes'  # Optional: adds a title to the legend
#     # Grid is enabled by default in Plotly
# )

# # --- 4. Show the interactive plot ---
# fig.show()
