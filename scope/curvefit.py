import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
from scipy.optimize import curve_fit
import numpy as np

warnings.filterwarnings('ignore')

# Define the input files, output files, number of columns, column names, and other parameters
N = 1  # select every Nth row
M = 1  # number of rows to skip

INPUT_FILE1 = "psd-bandwidth-2.csv"
NUM_COLUMNS_1 = 3  # number of columns (2 or 3)
COLUMN_TO_ANALYZE_1 = 'CH2'  # column to analyze

INPUT_FILE2 = "scopetest-laser-x-sum.csv"
NUM_COLUMNS_2 = 3  # number of columns (2 or 3)
COLUMN_TO_ANALYZE_2 = 'CH1'  # column to analyze

NAME = "Frequency Response of Thorlabs PSD PDP90A"

TRIM_DATA = False
OUTPUT_FILE1 = "trimmed_" + INPUT_FILE1
OUTPUT_FILE2 = "trimmed_" + INPUT_FILE2

REMOVE_OFFSET = False  # to remove any offset in channel data

# Define the reciprocal-like function
def reciprocal_func(x, a, b):
    return a / (x + b)

# Function to read a csv file, process the data, and plot the signal
def filter_and_plot(INPUT_FILE, OUTPUT_FILE, num_columns, column_to_analyse, label):
    if num_columns == 2:
        columns = ['Second', 'CH1']  # column names for 2-column CSV
    elif num_columns == 3:
        columns = ['Second', 'CH1', 'CH2']  # column names for 3-column CSV
    else:
        print("Invalid number of columns. Please enter 2 or 3.")
        exit()
    # Read the csv file, skipping the first M rows and setting the column names
    df = pd.read_csv(INPUT_FILE, skiprows=range(1, M + 1), names=columns)

    # Convert all columns to numeric, errors='coerce' will set non-numeric values to NaN
    for column in columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')

    # Drop rows with NaN values
    df = df.dropna()

    # Select every Nth row
    df = df.iloc[::N, :]
    
    # #SET X AXIS 1000x LARGER
    # df.iloc[:, 0] = df.iloc[:, 0] * 1000

    # Remove the offset of the signal if the REMOVE_OFFSET option is set to True
    if REMOVE_OFFSET:
        df[column_to_analyse] = df[column_to_analyse] - df[column_to_analyse].mean()
    
    # Save the trimmed dataframe to a new csv file
    df.to_csv(OUTPUT_FILE, index=False)

    # Plot the signal
    plt.plot(df[columns[0]], df[column_to_analyse], label=label)

    # Perform curve fitting
    popt, pcov = curve_fit(reciprocal_func, df[columns[0]], df[column_to_analyse])
    print(popt)
    fitted_y = reciprocal_func(df[columns[0]], *popt)

    # Plot the fitted curve
    plt.plot(df[columns[0]], fitted_y, label=rf'Fit of $V_{{pk}}=\frac{{a}}{{f+b}}$ with parameters [a,b] = [{popt[0]:.1f}, {popt[1]:.1f}]')

# Process and plot the signals from the two csv files
filter_and_plot(INPUT_FILE1, OUTPUT_FILE1, NUM_COLUMNS_1, COLUMN_TO_ANALYZE_1, f'Peak-Peak Response')
# filter_and_plot(INPUT_FILE2, OUTPUT_FILE2, NUM_COLUMNS_2, COLUMN_TO_ANALYZE_2, f'Laser On')

plt.title(f'{NAME}')

# plt.xlabel("Seconds")
# if REMOVE_OFFSET:
#     plt.ylabel('Voltage (Offset Normalised)')
# else:
#     plt.ylabel('Voltage')
# plt.legend()

plt.xlabel("Frequency (kHz)")
plt.axhline(y=0.92, color='red', linestyle='--', label='True Peak to Peak Variation.')
plt.xscale("log")
plt.yscale("log")
plt.ylabel('Pk-Pk (V)')

# Create the figures directory if it doesn't exist
if not os.path.exists('figures'):
    os.makedirs('figures')
    
plt.legend()

# Save the figure in the figures directory
plt.savefig(f'figures/{NAME}.png')

if TRIM_DATA:
    # Print success messages
    print(f"The trimmed dataframe was successfully saved to {OUTPUT_FILE1}.")
    print(f"The trimmed dataframe was successfully saved to {OUTPUT_FILE2}.")

plt.legend()
plt.show()
