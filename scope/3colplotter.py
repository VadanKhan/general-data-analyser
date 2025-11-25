import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# --- USER CONFIGURATION SECTION ---
# ==========================================

# 1. File Paths
DATA_FILENAME = "triangularity results.csv"      # Name of your .csv file
DATA_FOLDER = Path("data/")        # Folder where csv is located
FIGURES_FOLDER = Path("figures/")  # Folder to save the image
OUTPUT_NAME = "triangularity_tbr_sweep" # Name of the saved image file

# 2. Data Loading Settings
DELIMITER = ","                    # Separator (use ',' for csv, '\t' for tab-separated)
SKIP_HEADER = 1                    # Number of header rows to skip in the CSV file

# 3. Plot Labels and Titles
PLOT_TITLE = None
X_LABEL = "Triangularity"
Y_LABEL = "TBR"
LEGEND_LABEL = None

# 4. Visual Settings
MARKER_STYLE = 'o'                 # 'o' for circles, 's' for squares, '^' for triangles
LINE_STYLE = '-'                   # '-' for solid, '--' for dashed, '' for no line
COLOR = 'black'                # Hex code or color name (e.g., 'blue', '#FF5733')
CAP_SIZE = 3                       # Size of the horizontal caps on error bars
Y_SCALE = 'linear'                 # Options: 'linear', 'log'
X_SCALE = 'linear'                 # Options: 'linear', 'log'

# 5. Axis Limits (Set to None to let Matplotlib decide automatically)
X_LIMITS = None                    # Example: (0, 100) or None
Y_LIMITS = None                    # Example: (1e-3, 1) or None

# ==========================================
# --- END CONFIGURATION ---
# ==========================================

def main():
    # Ensure output directory exists
    FIGURES_FOLDER.mkdir(parents=True, exist_ok=True)

    try:
        # Construct full path
        file_path = DATA_FOLDER / DATA_FILENAME
        
        print(f"Loading data from: {file_path}")
        
        # Load Data
        # Assuming columns are: X, Y, Y-Error
        data = np.genfromtxt(file_path, dtype='float', delimiter=DELIMITER, skip_header=SKIP_HEADER)
        
        # Handle case where 1D array is returned (single row of data)
        if data.ndim == 1:
            data = data.reshape(1, -1)

        x = data[:, 0]
        y = data[:, 1]
        y_err = data[:, 2]

    except (IOError, FileNotFoundError) as e:
        print(f"\n[!] Error: Could not find or read '{DATA_FILENAME}'.")
        print(f"    Ensure the file exists in the '{DATA_FOLDER}' directory.")
        return
    except IndexError:
        print(f"\n[!] Error: The data file does not appear to have 3 columns.")
        print(f"    Found shape: {data.shape if 'data' in locals() else 'Unknown'}")
        return

    # Create Plot
    plt.figure(figsize=(8, 6)) # Optional: Set figure size (width, height)

    # Use errorbar plot
    plt.errorbar(
        x, y, 
        yerr=y_err, 
        fmt=MARKER_STYLE, # Combines marker and line style
        color=COLOR, 
        ecolor='black',     # Color of the error bars themselves
        elinewidth=1.5,     # Thickness of error bar lines
        capsize=CAP_SIZE,   # Adds the little horizontal caps
        label=LEGEND_LABEL,
        alpha=0.8           # Transparency
    )

    # Apply Settings
    plt.title(PLOT_TITLE, fontsize=14)
    plt.xlabel(X_LABEL, fontsize=12)
    plt.ylabel(Y_LABEL, fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.6) # 'both' works well for log scales
    
    # Scales
    plt.xscale(X_SCALE)
    plt.yscale(Y_SCALE)

    # Limits
    if X_LIMITS:
        plt.xlim(X_LIMITS)
    if Y_LIMITS:
        plt.ylim(Y_LIMITS)

    plt.legend(loc='best')

    # Save Figure
    save_path = FIGURES_FOLDER / f'{OUTPUT_NAME}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved successfully to: {save_path}")

    # Show Plot
    plt.show()

if __name__ == "__main__":
    main()