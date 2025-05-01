import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_excel("final data set.xlsx")

# Convert columns to appropriate types
df['ageOFocc'] = pd.to_numeric(df['ageOFocc'], errors='coerce')
df['sex'] = df['sex'].astype(str)
df['frontal'] = df['frontal'].astype(str)
df['airbag'] = df['airbag'].replace({'airbag': 'airbags deployed', 'none': 'airbags not deployed'}).astype(str)
df['dead'] = df['dead'].astype(str)
df['seatbelt'] = df['seatbelt'].replace({'belted': 'seatbelt worn', 'none': 'seatbelt not worn'}).astype(str)

# Print original statistics
print(f"Original dataset size: {len(df)} rows")
original_death_rate = (df['dead'] == 'dead').mean() * 100
print(f"Original death rate: {original_death_rate:.2f}%")

# Calculate how many alive cases to remove to achieve target death rate
target_death_rate = 15  # Target 15% death rate
current_alive = (df['dead'] == 'alive').sum()
current_dead = (df['dead'] == 'dead').sum()
target_total = current_dead / (target_death_rate / 100)
alive_to_remove = int(current_alive - (target_total - current_dead))

print(f"\nNeed to remove {alive_to_remove} alive cases to achieve {target_death_rate}% death rate")

# Randomly select alive cases to remove
alive_mask = df['dead'] == 'alive'
rows_to_remove = np.random.choice(df[alive_mask].index, size=alive_to_remove, replace=False)

# Create new dataframe without the selected rows
modified_df = df.drop(rows_to_remove)

# Print new statistics
print(f"\nModified dataset size: {len(modified_df)} rows")
new_death_rate = (modified_df['dead'] == 'dead').mean() * 100
print(f"New death rate: {new_death_rate:.2f}%")

# Print distribution of cases in modified dataset
print("\nDistribution in modified dataset:")
print("\nSeatbelt usage:")
print(modified_df['seatbelt'].value_counts(normalize=True).mul(100).round(2))
print("\nSurvival status:")
print(modified_df['dead'].value_counts(normalize=True).mul(100).round(2))
print("\nAirbag deployment:")
print(modified_df['airbag'].value_counts(normalize=True).mul(100).round(2))

# Save the modified dataset to a new Excel file
modified_df.to_excel("increased_death_rate_dataset.xlsx", index=False)
print("\nModified dataset saved to 'increased_death_rate_dataset.xlsx'") 