import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Load and Clean Data ---
file_path = "universal.csv"

try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: File {file_path} not found.")
    exit()

# Fill missing values downwards
df['Commit ID'] = df['Commit ID'].ffill()
df['Input Image'] = df['Input Image'].ffill()

# Convert columns to numeric
df['Time (sec)'] = pd.to_numeric(df['Time (sec)'], errors='coerce')
df['Peak Memory (KB)'] = pd.to_numeric(df['Peak Memory (KB)'], errors='coerce')

# Drop invalid rows
df.dropna(subset=['Time (sec)', 'Peak Memory (KB)', 'Commit ID', 'Input Image'], inplace=True)

# Shorten Commit ID
df['Short Commit'] = df['Commit ID'].astype(str).str[:7]

# Define Commit Order (Top = First)
unique_commits = df['Short Commit'].unique()
commit_to_x = {commit: i for i, commit in enumerate(unique_commits)}
df['x_pos'] = df['Short Commit'].map(commit_to_x)

# --- Function to Plot (Scatter + Mean Line) ---
def plot_with_trend(data, y_column, title, ylabel, filename, image_filter=None):
    plt.figure(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Filter data
    if image_filter:
        plot_data = data[data['Input Image'] == image_filter].copy()
    else:
        plot_data = data.copy()

    # Get unique images for coloring
    unique_imgs = plot_data['Input Image'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_imgs)))

    for i, img in enumerate(unique_imgs):
        subset = plot_data[plot_data['Input Image'] == img]
        
        # --- 1. Calculate and Plot the Average Line (The Trend) ---
        # Group by x_pos to ensure we get the mean for each commit position
        means = subset.groupby('x_pos')[y_column].mean()
        
        plt.plot(
            means.index, 
            means.values, 
            color=colors[i], 
            linestyle='-', 
            linewidth=2, 
            marker='o',    # Marker for the average point
            markersize=6,
            label=f"{img} (Mean)"
        )

        # --- 2. Plot the Scatter Points (The Raw Data) ---
        # Add Jitter to X-axis
        jitter = np.random.uniform(-0.1, 0.1, size=len(subset))
        
        plt.scatter(
            subset['x_pos'] + jitter, 
            subset[y_column], 
            color=colors[i],
            alpha=0.3,     # More transparent to make the line pop
            s=40, 
            edgecolors='none', # No edge for background dots to keep it clean
            label=None     # Don't add raw points to legend to save space
        )

    # Formatting
    plt.title(title, fontsize=14)
    plt.xlabel('Commit ID', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    
    plt.xticks(ticks=range(len(unique_commits)), labels=unique_commits, rotation=45)
    
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Input Image')
    plt.tight_layout()
    
    # Save as SVG
    print(f"Saving {filename}...")
    plt.savefig(filename, format='svg')
    plt.close()

# --- 2. Generate and Save the 3 Plots ---

# Plot 1: Turtle Time
plot_with_trend(
    df, 
    y_column='Time (sec)', 
    title='Execution Time - turtle.jpg (Mean Line + Raw Data)', 
    ylabel='Time (sec)', 
    filename='turtle_time_trend.svg',
    image_filter='turtle.jpg'
)

# Plot 2: One Hill Time
plot_with_trend(
    df, 
    y_column='Time (sec)', 
    title='Execution Time - one_hill.jpg (Mean Line + Raw Data)', 
    ylabel='Time (sec)', 
    filename='one_hill_time_trend.svg',
    image_filter='one_hill.jpg'
)

# Plot 3: Max Memory
plot_with_trend(
    df, 
    y_column='Peak Memory (KB)', 
    title='Peak Memory Usage (Mean Line + Raw Data)', 
    ylabel='Peak Memory (KB)', 
    filename='peak_memory_trend.svg',
    image_filter=None
)

print("Done! SVG files with trend lines created.")
