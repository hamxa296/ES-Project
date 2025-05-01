import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_excel("increased_death_rate_dataset.xlsx")

# Convert columns to appropriate types
df['ageOFocc'] = pd.to_numeric(df['ageOFocc'], errors='coerce')
df['sex'] = df['sex'].astype(str)
df['frontal'] = df['frontal'].astype(str)
df['airbag'] = df['airbag'].replace({'airbag': 'airbags deployed', 'none': 'airbags not deployed'}).astype(str)
df['dead'] = df['dead'].astype(str)
df['seatbelt'] = df['seatbelt'].replace({'belted': 'seatbelt worn', 'none': 'seatbelt not worn'}).astype(str)
df['occRole'] = df['occRole'].astype(str)

# Print original statistics
print(f"Original dataset size: {len(df)} rows")
print("\nOriginal distribution of occupant roles:")
print(df['occRole'].value_counts(normalize=True).mul(100).round(2))

# Remove rows where occRole is 'pass'
drivers_only_df = df[df['occRole'] != 'pass']

# Print new statistics
print(f"\nModified dataset size: {len(drivers_only_df)} rows")
print(f"Removed {len(df) - len(drivers_only_df)} passenger rows")

# Print distribution of cases in modified dataset
print("\nDistribution in modified dataset (drivers only):")
print("\nSeatbelt usage:")
print(drivers_only_df['seatbelt'].value_counts(normalize=True).mul(100).round(2))
print("\nSurvival status:")
print(drivers_only_df['dead'].value_counts(normalize=True).mul(100).round(2))
print("\nAirbag deployment:")
print(drivers_only_df['airbag'].value_counts(normalize=True).mul(100).round(2))
print("\nOccupant roles:")
print(drivers_only_df['occRole'].value_counts(normalize=True).mul(100).round(2))

# Save the modified dataset to a new Excel file
drivers_only_df.to_excel("drivers_only_dataset.xlsx", index=False)
print("\nModified dataset saved to 'drivers_only_dataset.xlsx'") 