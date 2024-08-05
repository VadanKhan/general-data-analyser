import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
import os
import warnings

warnings.filterwarnings('ignore')

# Define the input files, output files, number of columns, column names, and other parameters
N = 1  # select every Nth row
M = 12  # number of rows to skip

INPUT_FILE1 = "CSV-POS-L.csv"
NUM_COLUMNS_1 = 2  # number of columns (2 or 3)
COLUMN_TO_ANALYZE_1 = 'CH1'  # column to analyze

INPUT_FILE2 = "CSV-SUM-L.csv"
NUM_COLUMNS_2 = 2  # number of columns (2 or 3)
COLUMN_TO_ANALYZE_2 = 'CH1'  # column to analyze

INPUT_FILE3 = "CSV-ELEC-L.csv"
NUM_COLUMNS_3 = 2  # number of columns (2 or 3)
COLUMN_TO_ANALYZE_3 = 'CH1'  # column to analyze

NAME = "SUM signal to Noise signal"

NORMALIZE_FFT = False  # set to True to normalize the FFT based on the peak non-zero frequency
REMOVE_OFFSET = True  # to remove any offset in channel data
NORMALISE_INPUT = True

# Function to read a csv file, process the data, and plot the FFT of the signal
def fft_and_plot(INPUT_FILE, num_columns, column_to_analyse, label):
    if num_columns == 2:
        COLUMNS = ['Second', 'CH1']  # column names for 2-column CSV
    elif num_columns == 3:
        COLUMNS = ['Second', 'CH1', 'CH2']  # column names for 3-column CSV
    else:
        print("Invalid number of columns. Please enter 2 or 3.")
        exit()

    # Read the csv file, skipping the first M rows and setting the column names
    df = pd.read_csv(INPUT_FILE, skiprows=range(1, M + 1), names=COLUMNS)

    # Convert all columns to numeric, errors='coerce' will set non-numeric values to NaN
    for column in COLUMNS:
        df[column] = pd.to_numeric(df[column], errors='coerce')

    # Drop rows with NaN values
    df = df.dropna()

    # Select every Nth row
    df = df.iloc[::N, :].reset_index(drop=True)

    if REMOVE_OFFSET:
        df[column_to_analyse] = df[column_to_analyse] - df[column_to_analyse].mean()
        
    # if NORMALISE_INPUT:
    #     df[column_to_analyse] = df[column_to_analyse]/abs(df[column_to_analyse].max())
        
    # Perform FFT
    yf = fft(df[column_to_analyse].to_numpy())
    xf = np.linspace(0.0, 1.0/(2.0*(df[COLUMNS[0]][1]-df[COLUMNS[0]][0])), df.shape[0]//2) #plot up to half the Nyquist Frequency

    # Find the peak frequency and its value, ignoring the 0 frequency value
    yf_abs = 2.0/df.shape[0] * np.abs(yf[0:df.shape[0]//2])
    peak_index = np.argmax(yf_abs[1:]) + 1  # add 1 to reset the index from the 2nd starting point (ignores 0 frequency)
    peak_frequency = xf[peak_index]
    peak_value = yf_abs[peak_index]

    # Normalize the FFT result if the NORMALIZE option is set to True
    if NORMALIZE_FFT:
        yf_abs = yf_abs / peak_value
        peak_value = 1

    # Plot the FFT result
    plt.plot(xf, yf_abs, label=f'{label} (Peak Freq: {peak_frequency:.2f} Hz)')
    plt.plot(peak_frequency, peak_value, 'rx', markersize=10)

# Process and plot the FFTs of the signals from the two csv files
# fft_and_plot(INPUT_FILE1, NUM_COLUMNS_1, COLUMN_TO_ANALYZE_1, 'X signal')
fft_and_plot(INPUT_FILE2, NUM_COLUMNS_2, COLUMN_TO_ANALYZE_2, 'SUM')
fft_and_plot(INPUT_FILE3, NUM_COLUMNS_3, COLUMN_TO_ANALYZE_3, 'Electronic Noise')

# Highlight the peak PSD frequency (15kHz) with a vertical line
plt.axvline(x=15000, color='gray', linestyle='--', label='PSD bandwidth peak (15kHz)')

# Highlight the Cutoff region, for which we can assume that frequencies above are just noise
NOISE_CUTOFF = 15000
plt.axvline(x=NOISE_CUTOFF, color='red', linestyle='--', label=f'Noise Cutoff Region ({NOISE_CUTOFF/1000}kHz)')
plt.fill_betweenx(plt.ylim(), NOISE_CUTOFF, plt.xlim()[1], color='red', alpha=0.1)  # add weak red shading


plt.title(f'FFT of the channels in {NAME}')
plt.xlabel('Frequency')
if NORMALIZE_FFT:
    plt.ylabel('Magnitude (Peak Normalised)')
else:
    plt.ylabel('Magnitude')
plt.xscale("log")
plt.xlim([1, 200000])  # start at 1 instead of 0
plt.yscale("log")
plt.ylim([10**-8, 10**-1])  # start at 0.0001 instead of 0
plt.legend(loc='lower left', fontsize="9")

# Create the figures directory if it doesn't exist
if not os.path.exists('figures'):
    os.makedirs('figures')

# Save the figure in the figures directory
plt.savefig(f'figures/{NAME}-fft.png')

plt.show()
