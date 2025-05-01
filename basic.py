import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from colorama import init, Fore, Style
import seaborn as sns

# Initialize colorama for colored output
init()

# Set the backend to 'TkAgg' to better handle multiple figures
plt.switch_backend('TkAgg')

# Function to create and show a figure
def create_figure(figsize=(10, 6)):
    plt.close('all')  # Close any existing figures
    return plt.figure(figsize=figsize)

# Function to print formatted table with more prominent headers
def print_formatted_table(df, title):
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{title}{Style.RESET_ALL}")
    print("-"*80)
    
    # Get the maximum width of index entries
    max_index_width = max(len(str(idx)) for idx in df.index)
    if isinstance(df.index, pd.MultiIndex):
        max_index_width = sum(max(len(str(idx)) for idx in level) for level in df.index.levels) + 4
    
    # Print column headers in bright cyan (more visible than blue) and capitalized
    headers = df.columns
    index_spacing = " " * max_index_width
    header_line = f"{index_spacing}  {Fore.CYAN}{Style.BRIGHT}"
    header_line += "  ".join(str(col).upper().ljust(15) for col in headers)
    header_line += Style.RESET_ALL
    print(header_line)
    
    # Print separator line
    print(f"{index_spacing}  " + "-"*((15 + 2) * len(headers)))
    
    # Print the data with capitalized index
    for idx in df.index:
        if isinstance(idx, tuple):
            # Handle MultiIndex - capitalize each part
            idx_str = "  ".join(str(i).upper().ljust(20) for i in idx)
        else:
            # Capitalize single index
            idx_str = str(idx).upper().ljust(max_index_width)
        
        data_line = idx_str + "  " + "  ".join(str(round(df.loc[idx, col], 2)).ljust(15) if isinstance(df.loc[idx, col], (float, int)) 
                                              else str(df.loc[idx, col]).upper().ljust(15) for col in headers)
        print(data_line)
    print()

def load_and_preprocess_data(file_path):
    # Load the Excel file
    df = pd.read_excel(file_path)
    
    # Convert columns to appropriate types and standardize string values
    df['ageOFocc'] = pd.to_numeric(df['ageOFocc'], errors='coerce')
    
    # Standardize all string columns using string methods
    string_columns = ['sex', 'frontal', 'airbag', 'dead', 'seatbelt']
    for col in string_columns:
        df[col] = df[col].astype(str).str.upper().str.strip()
    
    # Standardize specific values for airbag and seatbelt columns
    df['airbag'] = df['airbag'].replace({
        'AIRBAG': 'AIRBAGS DEPLOYED',
        'NONE': 'AIRBAGS NOT DEPLOYED'
    })
    
    df['seatbelt'] = df['seatbelt'].replace({
        'BELTED': 'SEATBELT WORN',
        'NONE': 'SEATBELT NOT WORN'
    })
    
    return df

# Load and preprocess the data
df = load_and_preprocess_data("drivers_only_dataset.xlsx")

# Check the first few rows to confirm the column names
print(df.head())

# Calculate average and variance for the 'ageOfOCC' column
average_age = df['ageOFocc'].mean()
variance_age = df['ageOFocc'].var()

print(f"Average age of occupants: {average_age}")
print(f"Variance of age of occupants: {variance_age}")

# -----------------------------------------
# Confidence Interval and Tolerance Interval Analysis
# -----------------------------------------

# Drop NaN values from 'ageOFocc' column
age_data = df['ageOFocc'].dropna()

# Step 1: Split data into 80% train and 20% test
train_data, test_data = train_test_split(age_data, test_size=0.2, random_state=42)

# Step 2: Calculate 95% Confidence Interval for the Mean
n = len(train_data)
mean = np.mean(train_data)
std_dev = np.std(train_data, ddof=1)  # sample standard deviation
sem = stats.sem(train_data)           # standard error of the mean

confidence_level = 0.95
mean_conf_interval = stats.t.interval(confidence_level, df=n-1, loc=mean, scale=sem)

print(f"\n95% Confidence Interval for the mean age: ({round(float(mean_conf_interval[0]), 3)}, {round(float(mean_conf_interval[1]), 3)})")

# Step 3: Calculate 95% Confidence Interval for the Variance
alpha = 1 - confidence_level
chi2_lower = stats.chi2.ppf(alpha/2, df=n-1)
chi2_upper = stats.chi2.ppf(1 - alpha/2, df=n-1)

sample_variance = np.var(train_data, ddof=1)

variance_conf_interval = (
    (n-1) * sample_variance / chi2_upper,
    (n-1) * sample_variance / chi2_lower
)

print(f"95% Confidence Interval for the variance of age: ({round(float(variance_conf_interval[0]), 3)}, {round(float(variance_conf_interval[1]), 3)})")

# Step 4: Calculate 95% Tolerance Interval
g = 0.95  # confidence level for tolerance interval
p = 0.95  # proportion of population to cover

# Approximate k factor using F-distribution
k = np.sqrt((n - 1) * (1 + 1/n) * stats.f.ppf(g, 1, n - 1) / (n - 1))
lower_tolerance = mean - k * std_dev
upper_tolerance = mean + k * std_dev

print(f"95% Tolerance Interval for age: ({round(float(lower_tolerance), 3)}, {round(float(upper_tolerance), 3)})")

# Step 5: Check proportion of test samples inside the Tolerance Interval
inside_count = ((test_data >= lower_tolerance) & (test_data <= upper_tolerance)).sum()
total_test = len(test_data)
proportion_inside = inside_count / total_test

print(f"Proportion of test data inside tolerance interval: {inside_count}/{total_test} = {proportion_inside:.2f}")

# Create pie charts for all categorical variables
# Airbag deployment pie chart
create_figure(figsize=(8, 6))
airbag_counts = df['airbag'].value_counts()
explode = [0.1 if i == airbag_counts.idxmax() else 0 for i in airbag_counts.index]
plt.pie(airbag_counts, labels=airbag_counts.index, autopct='%1.1f%%', startangle=90, explode=explode)
plt.title('Airbag Deployment Distribution', pad=20, fontsize=14)
plt.tight_layout()
plt.show()

# Death status pie chart
create_figure(figsize=(8, 6))
dead_counts = df['dead'].value_counts()
explode = [0.1 if i == dead_counts.idxmax() else 0 for i in dead_counts.index]
plt.pie(dead_counts, labels=dead_counts.index, autopct='%1.1f%%', startangle=90, explode=explode)
plt.title('Death Status Distribution', pad=20, fontsize=14)
plt.tight_layout()
plt.show()

# Seatbelt usage pie chart
create_figure(figsize=(8, 6))
seatbelt_counts = df['seatbelt'].value_counts()
explode = [0.1 if i == seatbelt_counts.idxmax() else 0 for i in seatbelt_counts.index]
plt.pie(seatbelt_counts, labels=seatbelt_counts.index, autopct='%1.1f%%', startangle=90, explode=explode)
plt.title('Seatbelt Usage Distribution', pad=20, fontsize=14)
plt.tight_layout()
plt.show()

# Create histogram for age
plt.figure(figsize=(10, 5))
plt.hist(df['ageOFocc'], bins=10, alpha=0.7, edgecolor='black')
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Calculate statistics using frequency distribution
def calculate_stats_from_freq(column):
    freq = df[column].value_counts()
    values = freq.index
    weights = freq.values
    
    # Check if the values are numeric
    if pd.api.types.is_numeric_dtype(df[column]):
        avg = sum(values * weights) / sum(weights)
        var = sum((values - avg) ** 2 * weights) / sum(weights)
        return avg, var
    else:
        # For categorical data, return mode and frequency distribution
        mode = values[0]  # Most frequent value
        total = sum(weights)
        distribution = {val: (count/total) for val, count in zip(values, weights)}
        return mode, distribution

# Calculate for age (numerical)
age_avg, age_var = calculate_stats_from_freq('ageOFocc')
print(f"\nAge Statistics using Frequency Distribution:")
print(f"Average: {age_avg}")
print(f"Variance: {age_var}")

# Calculate for sex (categorical)
sex_mode, sex_dist = calculate_stats_from_freq('sex')
print(f"\nSex Statistics using Frequency Distribution:")
print(f"Mode (Most common): {sex_mode}")
print("Distribution:")
for val, prop in sex_dist.items():
    print(f"{val}: {prop:.2%}")

# Calculate for frontal (categorical)
frontal_mode, frontal_dist = calculate_stats_from_freq('frontal')
print(f"\nFrontal Statistics using Frequency Distribution:")
print(f"Mode (Most common): {frontal_mode}")
print("Distribution:")
for val, prop in frontal_dist.items():
    print(f"{val}: {prop:.2%}")

# Calculate for airbag (categorical)
airbag_mode, airbag_dist = calculate_stats_from_freq('airbag')
print(f"\nAirbag Statistics using Frequency Distribution:")
print(f"Mode (Most common): {airbag_mode}")
print("Distribution:")
for val, prop in airbag_dist.items():
    print(f"{val}: {prop:.2%}")

# Calculate for death status (categorical)
dead_mode, dead_dist = calculate_stats_from_freq('dead')
print(f"\nDeath Status Statistics using Frequency Distribution:")
print(f"Mode (Most common): {dead_mode}")
print("Distribution:")
for val, prop in dead_dist.items():
    print(f"{val}: {prop:.2%}")

# Calculate for seatbelt (categorical)
seatbelt_mode, seatbelt_dist = calculate_stats_from_freq('seatbelt')
print(f"\nSeatbelt Statistics using Frequency Distribution:")
print(f"Mode (Most common): {seatbelt_mode}")
print("Distribution:")
for val, prop in seatbelt_dist.items():
    print(f"{val}: {prop:.2%}")

# Create contingency tables for analysis
# Airbag vs Death
airbag_death = pd.crosstab(df['airbag'], df['dead'])
airbag_death.columns = ['SURVIVED', 'DECEASED']
print_formatted_table(airbag_death, "CONTINGENCY TABLE: AIRBAG DEPLOYMENT VS DEATH STATUS")

# Seatbelt vs Death
seatbelt_death = pd.crosstab(df['seatbelt'], df['dead'])
seatbelt_death.columns = ['SURVIVED', 'DECEASED']
print_formatted_table(seatbelt_death, "CONTINGENCY TABLE: SEATBELT USAGE VS DEATH STATUS")

# Combined effect: Airbag and Seatbelt vs Death
airbag_seatbelt_death = pd.crosstab([df['airbag'], df['seatbelt']], df['dead'])
airbag_seatbelt_death.columns = ['SURVIVED', 'DECEASED']
print_formatted_table(airbag_seatbelt_death, "CONTINGENCY TABLE: COMBINED EFFECT (AIRBAG AND SEATBELT) VS DEATH STATUS")

# Print the actual index values
print("\nActual index values in airbag_seatbelt_death:")
for idx in airbag_seatbelt_death.index:
    print(idx)

# Combined effect visualization (Bar Chart)
plt.figure(figsize=(12, 6))
# Create labels using the actual values from the DataFrame
labels = []
for airbag, belt in airbag_seatbelt_death.index:
    # Format the labels to be more readable
    airbag_label = "Airbag Deployed" if "DEPLOYED" in airbag else "Airbag Not Deployed"
    belt_label = "Seatbelt Worn" if "WORN" in belt else "Seatbelt Not Worn"
    labels.append(f"{airbag_label}\n{belt_label}")

x = range(len(labels))
plt.bar(x, airbag_seatbelt_death['SURVIVED'])
plt.title('Survival Count by Airbag Deployment and Seatbelt Usage', pad=20)
plt.xlabel('Airbag Deployment and Seatbelt Usage', labelpad=10)
plt.ylabel('Number of Survivors', labelpad=10)
plt.xticks(x, labels, rotation=0)
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

# Stacked Bar Graph
create_figure(figsize=(10, 6))
# Group data by seatbelt and airbag
stacked_data = airbag_seatbelt_death.unstack(level=0)
# Ensure we have the correct number of columns
if len(stacked_data.columns) == 4:
    stacked_data.columns = ['Airbag Deployed - Survived', 'Airbag Deployed - Deceased', 
                          'Airbag Not Deployed - Survived', 'Airbag Not Deployed - Deceased']
stacked_data.index = ['Seatbelt Worn', 'Seatbelt Not Worn']

# Plot stacked bars
stacked_data.plot(kind='bar', stacked=True)
plt.title('Survival Count by Seatbelt Usage and Airbag Deployment', pad=20)
plt.xlabel('Seatbelt Usage', labelpad=10)
plt.ylabel('Number of Cases', labelpad=10)
plt.legend(title='Airbag and Survival Status', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

# Boxplot of Age by Survival Status
create_figure(figsize=(12, 6))
# Create subplots for different combinations
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Boxplot 1: Age by Survival Status
df.boxplot(column='ageOFocc', by='dead', ax=axes[0])
axes[0].set_title('Age Distribution by Survival Status')
axes[0].set_xlabel('Survival Status')
axes[0].set_ylabel('Age')

# Boxplot 2: Age by Survival Status and Seatbelt Usage
df.boxplot(column='ageOFocc', by=['dead', 'seatbelt'], ax=axes[1])
axes[1].set_title('Age Distribution by Survival Status and Seatbelt Usage')
axes[1].set_xlabel('Survival Status and Seatbelt Usage')
axes[1].set_ylabel('Age')
plt.tight_layout()
plt.show()

# Chi-Square Analysis Visualization
create_figure(figsize=(15, 5))
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Airbag vs Survival
airbag_survival = pd.crosstab(df['airbag'], df['dead'])
airbag_survival.plot(kind='bar', stacked=True, ax=axes[0])
axes[0].set_title('Airbag Deployment vs Survival Status')
axes[0].set_xlabel('Airbag Deployment')
axes[0].set_ylabel('Count')
axes[0].legend(title='Survival Status')

# Seatbelt vs Survival
seatbelt_survival = pd.crosstab(df['seatbelt'], df['dead'])
seatbelt_survival.plot(kind='bar', stacked=True, ax=axes[1])
axes[1].set_title('Seatbelt Usage vs Survival Status')
axes[1].set_xlabel('Seatbelt Usage')
axes[1].set_ylabel('Count')
axes[1].legend(title='Survival Status')

plt.tight_layout()
plt.show()

# Confidence Interval for Mean Age
create_figure(figsize=(6, 1.5))
mean_age = 38.17
ci_low = 37.432
ci_high = 39.142

plt.errorbar(mean_age, 0, xerr=[[mean_age - ci_low], [ci_high - mean_age]], fmt='o', color='blue', capsize=5)
plt.title('95% Confidence Interval for Mean Age')
plt.xlabel('Age')
plt.yticks([])
plt.grid(True)
plt.tight_layout()
plt.show()

# Confidence Interval for Variance
create_figure(figsize=(6, 1.5))
var = 317.586
ci_var_low = 295.683
ci_var_high = 338.738

plt.errorbar(var, 0, xerr=[[var - ci_var_low], [ci_var_high - var]], fmt='o', color='green', capsize=5)
plt.title('95% Confidence Interval for Age Variance')
plt.xlabel('Variance')
plt.yticks([])
plt.grid(True)
plt.tight_layout()
plt.show()

# Tolerance Interval Plot
create_figure(figsize=(8, 4))
plt.hist(df['ageOFocc'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(3.404, color='red', linestyle='--', label='Lower Tolerance Bound')
plt.axvline(73.17, color='red', linestyle='--', label='Upper Tolerance Bound')
plt.title('95% Tolerance Interval on Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.show()

# Expected vs Observed Frequencies Visualization
plt.figure(figsize=(15, 5))
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Airbag expected vs observed
chi2, p, dof, expected = stats.chi2_contingency(airbag_survival)
expected_df = pd.DataFrame(expected, index=airbag_survival.index, columns=airbag_survival.columns)
expected_df.plot(kind='bar', ax=axes[0])
axes[0].set_title('Expected Frequencies: Airbag vs Survival')
axes[0].set_xlabel('Airbag Deployment')
axes[0].set_ylabel('Expected Count')
axes[0].legend(title='Survival Status')

# Seatbelt expected vs observed
chi2, p, dof, expected = stats.chi2_contingency(seatbelt_survival)
expected_df = pd.DataFrame(expected, index=seatbelt_survival.index, columns=seatbelt_survival.columns)
expected_df.plot(kind='bar', ax=axes[1])
axes[1].set_title('Expected Frequencies: Seatbelt vs Survival')
axes[1].set_xlabel('Seatbelt Usage')
axes[1].set_ylabel('Expected Count')
axes[1].legend(title='Survival Status')

plt.tight_layout()
plt.show()

# Print header for dataset overview
print("\n" + "="*80)
print("DATASET OVERVIEW".center(80))
print("="*80 + "\n")

print("First few rows of the dataset:")
print("-"*40)
print(df.head().to_string())
print("\n")

# Basic Age Statistics
print("="*80)
print("BASIC AGE STATISTICS".center(80))
print("="*80)
print(f"Average age of occupants: {round(average_age, 3)} years")
print(f"Variance of age of occupants: {round(variance_age, 3)} years²\n")

# Age Statistics using Frequency Distribution
print("="*80)
print("AGE STATISTICS USING FREQUENCY DISTRIBUTION".center(80))
print("="*80)
print(f"Average: {round(age_avg, 3)} years")
print(f"Variance: {round(age_var, 3)} years²\n")

# Sex Statistics
print("="*80)
print("SEX DISTRIBUTION".center(80))
print("="*80)
print(f"Mode (Most common): {sex_mode}")
print("Distribution:")
for val, prop in sex_dist.items():
    print(f"{val}: {round(prop*100, 3)}%")
print()

# Frontal Statistics
print("="*80)
print("FRONTAL COLLISION DISTRIBUTION".center(80))
print("="*80)
print(f"Mode (Most common): {frontal_mode}")
print("Distribution:")
for val, prop in frontal_dist.items():
    print(f"{val}: {round(prop*100, 3)}%")
print()

# Airbag Statistics
print("="*80)
print("AIRBAG DEPLOYMENT DISTRIBUTION".center(80))
print("="*80)
print(f"Mode (Most common): {airbag_mode}")
print("Distribution:")
for val, prop in airbag_dist.items():
    print(f"{val}: {round(prop*100, 3)}%")
print()

# Death Status Statistics
print("="*80)
print("SURVIVAL STATUS DISTRIBUTION".center(80))
print("="*80)
print(f"Mode (Most common): {dead_mode}")
print("Distribution:")
for val, prop in dead_dist.items():
    print(f"{val}: {round(prop*100, 3)}%")
print()

# Seatbelt Statistics
print("="*80)
print("SEATBELT USAGE DISTRIBUTION".center(80))
print("="*80)
print(f"Mode (Most common): {seatbelt_mode}")
print("Distribution:")
for val, prop in seatbelt_dist.items():
    print(f"{val}: {round(prop*100, 3)}%")
print()

# Confidence and Tolerance Intervals
print("="*80)
print("CONFIDENCE AND TOLERANCE INTERVALS ANALYSIS".center(80))
print("="*80)
print("\nConfidence Intervals (95%):")
print("-"*40)
print(f"Mean Age: ({round(float(mean_conf_interval[0]), 3)}, {round(float(mean_conf_interval[1]), 3)})")
print(f"Variance: ({round(float(variance_conf_interval[0]), 3)}, {round(float(variance_conf_interval[1]), 3)})")

print("\nTolerance Interval (95/95):")
print("-"*40)
print(f"Age Range: ({round(float(lower_tolerance), 3)}, {round(float(upper_tolerance), 3)})")
print(f"Validation: {inside_count} out of {total_test} test samples ({round(proportion_inside*100, 3)}%) fall within the interval\n")

# Frequency Distribution Statistics
print("="*80)
print("FREQUENCY DISTRIBUTION STATISTICS".center(80))
print("="*80)

print("\nAge Statistics:")
print("-"*40)
print(f"Average: {round(age_avg, 3)} years")
print(f"Variance: {round(age_var, 3)} years²")

print("\nSex Distribution:")
print("-"*40)
print(f"Most common: {sex_mode}")
for val, prop in sex_dist.items():
    print(f"{val}: {round(prop*100, 3)}%")

print("\nFrontal Collision Distribution:")
print("-"*40)
print(f"Most common: {frontal_mode}")
for val, prop in frontal_dist.items():
    print(f"{val}: {round(prop*100, 3)}%")

print("\nAirbag Deployment Distribution:")
print("-"*40)
print(f"Most common: {airbag_mode}")
for val, prop in airbag_dist.items():
    print(f"{val}: {round(prop*100, 3)}%")

print("\nSurvival Status Distribution:")
print("-"*40)
print(f"Most common: {dead_mode}")
for val, prop in dead_dist.items():
    print(f"{val}: {round(prop*100, 3)}%")

print("\nSeatbelt Usage Distribution:")
print("-"*40)
print(f"Most common: {seatbelt_mode}")
for val, prop in seatbelt_dist.items():
    print(f"{val}: {round(prop*100, 3)}%")

print("\n" + "="*80)
print("ANALYSIS COMPLETE".center(80))
print("="*80)


