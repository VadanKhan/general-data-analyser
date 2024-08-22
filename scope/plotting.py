import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
import numpy as np

warnings.filterwarnings('ignore')

# Define the input files, output files, number of columns, column names, and other parameters
N = 100  # select every Nth row
M = 12  # number of rows to skip

INPUT_FILE1 = "CSVDREAM5.csv"
NUM_COLUMNS_1 = 3  # number of columns (2 or 3)
COLUMN_TO_ANALYZE_1 = 'CH1'  # column to analyze

INPUT_FILE2 = "CSVESPDELAY.csv"
NUM_COLUMNS_2 = 3  # number of columns (2 or 3)
COLUMN_TO_ANALYZE_2 = 'CH2'  # column to analyze

NAME = "Beam Feedback Demonstration on 100Hz Noise"

TRIM_DATA = False
OUTPUT_FILE1 = "trimmed_" + INPUT_FILE1
OUTPUT_FILE2 = "trimmed_" + INPUT_FILE2

REMOVE_OFFSET = False  # to remove any offset in channel data

RECALIBRATE_VOLTAGE = True
BASE_REST_VOLTAGE = 0.180  # experimentally set constant
VRANGE = 2  #experimental constant
DIODE_SIZE = 0.01 * 1000000  # CONVERTed TO MICROMETERS (for thorlabs psd)


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

    # Remove the offset of the signal if the REMOVE_OFFSET option is set to True
    if REMOVE_OFFSET:
        df[column_to_analyse] = df[column_to_analyse] - df[column_to_analyse].mean()
        
    # Recalibrate voltage to x position in micrometers
    if RECALIBRATE_VOLTAGE: 
        df[column_to_analyse] = df[column_to_analyse] - BASE_REST_VOLTAGE
        df[column_to_analyse] = df[column_to_analyse] * (DIODE_SIZE) / VRANGE 
        
    
    # Save the trimmed dataframe to a new csv file
    df.to_csv(OUTPUT_FILE, index=False)

    # Plot the signal
    plt.scatter(df[columns[0]], df[column_to_analyse], label=label, alpha=0.01, s=20, marker='o')
    
    return df

# Process and plot the signals from the two csv files
df1 = filter_and_plot(INPUT_FILE1, OUTPUT_FILE1, NUM_COLUMNS_1, COLUMN_TO_ANALYZE_1, f'Beam Position PSD Reading')
# filter_and_plot(INPUT_FILE2, OUTPUT_FILE2, NUM_COLUMNS_2, COLUMN_TO_ANALYZE_2, f'DAC output Signal')

plt.title(f'{NAME}')

plt.xlabel("Seconds")
if REMOVE_OFFSET:
    plt.ylabel('Voltage (Offset Normalised)')
else:
    plt.ylabel(r'Beam Position ($\mu$m)')

# Plotting the horizontal line
plt.axhline(y=0, color='red', linestyle='-', label='Desired Beam Position')

# Plotting the vertical lines
x_positions = [-4.85, -3.45, -1.8, -0.4, 1.1, 2.6, 4.1, 5.6]
for x in x_positions:
    plt.axvline(x=x, color='grey', linestyle='--')
    
# Annotating "OFF" and "ON"
annotations = ["ON", "OFF"]
for i in range(len(x_positions) - 1):
    x_mid = (x_positions[i] + x_positions[i + 1]) / 2
    plt.text(x_mid, 400, annotations[i % 2], ha='center')
# Adding "OFF" for the last two sections
plt.text(x_positions[0] - 0.75, 400, "OFF", ha='center')
plt.text(x_positions[-1] + 0.75, 400, "OFF", ha='center')

# Extracting the data for standard deviation calculation
time = df1['Second']
data = df1[COLUMN_TO_ANALYZE_1]

# Calculating standard deviation for the blue section (OFF)
blue_section_data = [value for value, t in zip(data, time) if -3.45 <= t <= -1.8 and -121 <= value <= 381]
blue_section_mean = np.mean(blue_section_data)
blue_std_dev = np.std(blue_section_data)
print(f"Standard Deviation for Blue Section: {blue_std_dev}")

# Calculating standard deviation for the green section (ON)
green_section_data = [value for value, t in zip(data, time) if -1.8 <= t <= -0.4 and -121 <= value <= 121] # 
green_section_mean = np.mean(green_section_data)
green_std_dev = np.std(green_section_data)
print(f"Standard Deviation for Green Section: {green_std_dev}")

ratio = green_std_dev / blue_std_dev

print(rf"ON std / OFF std = {ratio}")

# Calculate the ranges for blue and green sections using 95.45% of the data points
def calculate_range(data):
    sorted_data = np.sort(data)
    n_points = len(sorted_data)
    lower_index = int((1 - 0.9545) / 2 * n_points)
    upper_index = int((1 + 0.9545) / 2 * n_points) - 1
    return sorted_data[lower_index], sorted_data[upper_index]

blue_range_min, blue_range_max = calculate_range(blue_section_data)
green_range_min, green_range_max = calculate_range(green_section_data)

print(f"OFF Range: {blue_range_min} to {blue_range_max}")
print(f"ON Range: {green_range_min} to {green_range_max}")

blue_range = blue_range_max - blue_range_min
green_range = green_range_max - green_range_min
ratio2 = green_range / blue_range
print(rf"ON range / OFF range = {ratio2}")

# Adding annotations for the 3rd and 4th segments
# Blue section (OFF)
plt.hlines(y=[blue_range_min, blue_range_max], xmin=-3.45, xmax=-1.8, color='blue', 
           label=rf'Feedback OFF Range: {blue_range_min:.1f}$\mu$m to {blue_range_max:.1f}$\mu$m ($\sigma_{{off}}=${blue_std_dev:.1f})')
plt.annotate('', xy=(-2.2, blue_range_min), xytext=(-2.2, blue_range_max), arrowprops=dict(arrowstyle='<->', color='blue'))
plt.text(-1.4, blue_range_max + 20, r'OFF 2$\sigma$', ha='center', color='blue')

# Green section (ON)
plt.hlines(y=[green_range_min, green_range_max], xmin=-1.8, xmax=-0.4, color='limegreen', 
           label=rf'Feedback ON Range: {green_range_min:.1f}$\mu$m to {green_range_max:.1f}$\mu$m ($\sigma_{{on}}=${green_std_dev:.1f}, {ratio*100:.1f}% of $\sigma_{{off}}$)')
plt.annotate('', xy=(-0.8, green_range_min), xytext=(-0.8, green_range_max), arrowprops=dict(arrowstyle='<->', color='limegreen'))
plt.text(-0.8, green_range_min - 60, r'ON 2$\sigma$', ha='center', color='green')

# plt.xlabel("Frequency (Hz)")
# plt.xlim([50, 10000])
# plt.ylim([9, 1000])
# plt.xscale('log')
# plt.ylabel(r'Beam Motion ($\mu$m)')
# plt.yscale('log')

# plt.axvline(x=NOISE_CUTOFF, color='grey', linestyle='--', label='Manufacturer Stated Limit (15kHz)')
# plt.fill_betweenx(plt.ylim(), NOISE_CUTOFF, plt.xlim(), color='gray', alpha=0.1)

# CALCULATED_POSITION = 360
# UNCERTAINTY_POSITION = 100
# plt.fill_between(plt.xlim(), CALCULATED_POSITION - UNCERTAINTY_POSITION, CALCULATED_POSITION + UNCERTAINTY_POSITION, 
#                  color='red', alpha=0.15)
# plt.axhline(y=CALCULATED_POSITION, color='red', linestyle='--', label=r'Specification Expected Amplitude: 360$\mu$m')
# plt.plot(900, 700, '.', label='Resonance 1: 900Hz', markersize=10, markeredgewidth=0.01)  # Corrected the fmt argument
# plt.plot(2600, 810, '.', label='Resonance 2: 2600Hz', markersize=10, markeredgewidth=0.01)  # Corrected the fmt argument
# plt.plot(3500, 90, 'X', label='First -5dB Point: 3500Hz', markersize=10, markeredgewidth=0.01)  # Corrected the fmt argument
# plt.axhline(y=10, color='gray', linestyle='--', label='Electronic Noise in Position Reading')

plt.legend(loc='center left',bbox_to_anchor=(0.0, 0.12), fontsize=8)
# plt.legend()


# Create the figures directory if it doesn't exist
if not os.path.exists('figures'):
    os.makedirs('figures')
    
# Save the figure in the figures directory
plt.savefig(f'figures/{NAME}.png')

if TRIM_DATA:
    # Save the figure in the figures directory
    plt.savefig(f'figures/{NAME}.png')
    # Print success messages
    print(f"The trimmed dataframe was successfully saved to {OUTPUT_FILE1}.")
    print(f"The trimmed dataframe was successfully saved to {OUTPUT_FILE2}.")

plt.show()
