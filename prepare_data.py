import pandas as pd
import os

# --- Configuration ---
DATA_FILES = {
    '0_blink_data.csv': 0, 
    '1_blink_data.csv': 1,
    '2_blink_data.csv': 2
}
OUTPUT_FILENAME = 'eog_processed_data.csv'

# --- Main Script ---
all_dataframes = []
print("Starting data processing...")

for filename, label in DATA_FILES.items():
    if os.path.exists(filename):
        print(f"Processing file: {filename} with label {label}...")
        
        # Read the CSV file
        df = pd.read_csv(filename)
        
        # --- THIS IS THE CRITICAL FIX ---
        # Force the 'signal' column to be numeric.
        # Any text (like a header) will become 'NaN' (Not a Number).
        df['signal'] = pd.to_numeric(df['signal'], errors='coerce')
        
        # Drop any rows where the conversion failed
        df.dropna(subset=['signal'], inplace=True)
        # --- END OF FIX ---

        # Add the new 'label' column
        df['label'] = label
        all_dataframes.append(df)
    else:
        print(f"Warning: File '{filename}' not found. Skipping.")

if not all_dataframes:
    print("Error: No data files were found. Exiting.")
else:
    # Combine all the individual dataframes into one
    combined_df = pd.concat(all_dataframes, ignore_index=True)

    # Shuffle the dataset randomly
    shuffled_df = combined_df.sample(frac=1).reset_index(drop=True)

    # Save the final, processed dataset to a new CSV file
    shuffled_df.to_csv(OUTPUT_FILENAME, index=False)

    print("\nProcessing complete!")
    print(f"Final cleaned and shuffled dataset saved as '{OUTPUT_FILENAME}'")
    print(f"Total rows: {len(shuffled_df)}")
    print("Label distribution:")
    print(shuffled_df['label'].value_counts())

import matplotlib.pyplot as plt
import random

# After saving the shuffled CSV, add this plotting code
print("\nPlotting sample windows from each class...")

# Group data by label
grouped = shuffled_df.groupby('label')
window_size = 75 # Your window size

# Create a plot with 3 rows
fig, axs = plt.subplots(3, 3, figsize=(15, 10))
fig.suptitle('Random Sample Windows from Each Class')

for i, (label, group) in enumerate(grouped):
    for j in range(3): # Plot 3 samples for each class
        # Pick a random starting point in the group's data
        start_index = random.choice(group.index[:-window_size])
        
        # Get the window
        sample_window = shuffled_df.loc[start_index:start_index + window_size - 1]
        
        axs[i, j].plot(sample_window['time'], sample_window['signal'])
        axs[i, j].set_title(f'Class {label} - Sample {j+1}')
        axs[i, j].set_ylim(-1.5, 1.5) # Set consistent y-axis limits

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()