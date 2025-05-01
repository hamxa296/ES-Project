import pandas as pd
import numpy as np

# Load the original dataset
df = pd.read_excel("final data set.xlsx")

# Convert columns to appropriate types
df['ageOFocc'] = pd.to_numeric(df['ageOFocc'], errors='coerce')
df['sex'] = df['sex'].astype(str)
df['frontal'] = df['frontal'].astype(str)
df['airbag'] = df['airbag'].replace({'airbag': 'airbags deployed', 'none': 'airbags not deployed'}).astype(str)
df['dead'] = df['dead'].astype(str)
df['seatbelt'] = df['seatbelt'].replace({'belted': 'seatbelt worn', 'none': 'seatbelt not worn'}).astype(str)

# Print original dataset size
print(f"Original dataset size: {len(df)} rows")

# Identify rows where occupant is alive and wearing seatbelt
alive_belted_mask = (df['dead'] == 'alive') & (df['seatbelt'] == 'seatbelt worn')

# Count how many such rows exist
alive_belted_count = alive_belted_mask.sum()
print(f"Number of rows where occupant is alive and wearing seatbelt: {alive_belted_count}")

# Randomly select 3000 rows to remove from these cases
rows_to_remove = np.random.choice(df[alive_belted_mask].index, size=1000, replace=False)

# Create new dataframe without the selected rows
reduced_df = df.drop(rows_to_remove)

# Print new dataset size
print(f"Reduced dataset size: {len(reduced_df)} rows")

# Save the reduced dataset to a new Excel file
reduced_df.to_excel("reduced_dataset_new.xlsx", index=False)
print("Reduced dataset saved to 'reduced_dataset_new.xlsx'")

# Print some statistics about the reduction
print("\nStatistics about the reduction:")
print(f"Rows removed: {len(df) - len(reduced_df)}")
print(f"Percentage reduction: {((len(df) - len(reduced_df)) / len(df) * 100):.2f}%")

# Print distribution of cases in reduced dataset
print("\nDistribution in reduced dataset:")
print("\nSeatbelt usage:")
print(reduced_df['seatbelt'].value_counts(normalize=True).mul(100).round(2))
print("\nSurvival status:")
print(reduced_df['dead'].value_counts(normalize=True).mul(100).round(2)) 