#Saad H. Ellahie, Reg #2024545, Hamza Arif, Reg #2024206

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from colorama import init, Fore, Style
import seaborn as sns
from scipy.stats import chi2_contingency
from statsmodels.graphics.mosaicplot import mosaic

# Initialize colorama for colored output
init()


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

# Print dataset overview at the top
print("\n" + "="*80)
print("DATASET OVERVIEW".center(80))
print("="*80 + "\n")

print("First few rows of the dataset:")
print("-"*40)
print(df.head().to_string())
print("\n")

# Calculate average and variance for the 'ageOfOCC' column manually
# Calculate mean manually
sum_age = 0
count = 0
for age in df['ageOFocc']:
    if not pd.isna(age):  # Skip NaN values
        sum_age += age
        count += 1
average_age = sum_age / count

# Calculate variance manually
sum_squared_diff = 0
for age in df['ageOFocc']:
    if not pd.isna(age):  # Skip NaN values
        sum_squared_diff += (age - average_age) ** 2
variance_age = sum_squared_diff / (count - 1)  # Using sample variance formula (n-1)

# Print header for dataset overview
print("\n" + "="*80)
print("BASIC AGE STATISTICS".center(80))
print("="*80)
print(f"Average age of occupants: {round(average_age, 3)} years")
print(f"Variance of age of occupants: {round(variance_age, 3)} years²\n")


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


# Step 3: Calculate 95% Confidence Interval for the Variance
alpha = 1 - confidence_level
chi2_lower = stats.chi2.ppf(alpha/2, df=n-1)
chi2_upper = stats.chi2.ppf(1 - alpha/2, df=n-1)

sample_variance = np.var(train_data, ddof=1)

variance_conf_interval = (
    (n-1) * sample_variance / chi2_upper,
    (n-1) * sample_variance / chi2_lower
)


# Step 4: Calculate 95% Tolerance Interval
g = 0.95  # confidence level for tolerance interval
p = 0.95  # proportion of population to cover

# Approximate k factor using F-distribution
k = np.sqrt((n - 1) * (1 + 1/n) * stats.f.ppf(g, 1, n - 1) / (n - 1))
lower_tolerance = mean - k * std_dev
upper_tolerance = mean + k * std_dev


# Step 5: Check proportion of test samples inside the Tolerance Interval
inside_count = ((test_data >= lower_tolerance) & (test_data <= upper_tolerance)).sum()
total_test = len(test_data)
proportion_inside = inside_count / total_test


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

# Calculate for sex (categorical)
sex_mode, sex_dist = calculate_stats_from_freq('sex')

# Calculate for frontal (categorical)
frontal_mode, frontal_dist = calculate_stats_from_freq('frontal')

# Calculate for airbag (categorical)
airbag_mode, airbag_dist = calculate_stats_from_freq('airbag')

# Calculate for death status (categorical)
dead_mode, dead_dist = calculate_stats_from_freq('dead')

# Calculate for seatbelt (categorical)
seatbelt_mode, seatbelt_dist = calculate_stats_from_freq('seatbelt')

# Age Statistics using Frequency Distribution
print("="*80)
print("AGE STATISTICS USING FREQUENCY DISTRIBUTION".center(80))
print("="*80)
print(f"Average: {round(age_avg, 3)} years")
print(f"Variance: {round(age_var, 3)} years²\n")

# Confidence and Tolerance Intervals
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

print("="*80)
print("CHI-SQUARE TEST FOR AIRBAG DEPLOYMENT AND FATALITY".center(80))
print("="*80)

# Create a contingency table
contingency_table = pd.crosstab(df['airbag'], df['dead'])

# Perform the Chi-Squared test
chi2, p_value, dof, expected = chi2_contingency(contingency_table)

# Output the results
print("\nContingency Table:")
print(contingency_table)
print(f"\nChi-Squared Statistic: {round(chi2, 3)}")
print(f"Degrees of Freedom: {dof}")
print("\nExpected Counts:")
print(pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns))
print(f"\nP-value: {p_value}")

# Determine if the variables are independent
alpha = 0.05  # Significance level
if p_value < alpha:
    print("\nReject the null hypothesis: Airbag deployment and fatality are dependent.")
else:
    print("\nFail to reject the null hypothesis: Airbag deployment and fatality are independent.")

# Combined effect visualization (Bar Chart)
plt.figure(figsize=(12, 6))
# Create a new DataFrame for the visualization
airbag_seatbelt_survival = pd.crosstab([df['airbag'], df['seatbelt']], df['dead'])
airbag_seatbelt_survival.columns = ['SURVIVED', 'DECEASED']

# Create labels using the actual values from the DataFrame
labels = []
for airbag, belt in airbag_seatbelt_survival.index:
    # Format the labels to be more readable
    airbag_label = "Airbag Deployed" if "DEPLOYED" in airbag else "Airbag Not Deployed"
    belt_label = "Seatbelt Worn" if "WORN" in belt else "Seatbelt Not Worn"
    labels.append(f"{airbag_label}\n{belt_label}")

x = range(len(labels))
plt.bar(x, airbag_seatbelt_survival['SURVIVED'])
plt.title('Survival Count by Airbag Deployment and Seatbelt Usage', pad=20)
plt.xlabel('Airbag Deployment and Seatbelt Usage', labelpad=10)
plt.ylabel('Number of Survivors', labelpad=10)
plt.xticks(x, labels, rotation=0)
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

# Stacked Bar Graph
plt.figure(figsize=(10, 6))
# Group data by seatbelt and airbag
stacked_data = airbag_seatbelt_survival.unstack(level=0)
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
plt.figure(figsize=(12, 6))
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
plt.figure(figsize=(15, 5))
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
plt.figure(figsize=(6, 1.5))
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
plt.figure(figsize=(6, 1.5))
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
plt.figure(figsize=(8, 4))
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

airbag_survivors_counts = df[df['dead'] == 'ALIVE']['airbag'].value_counts()
plt.figure(figsize=(8, 6))
plt.pie(airbag_survivors_counts, startangle=90, colors=['#4CAF50', '#81C784'])
plt.title('Airbag Deployment Among Survivors')
plt.legend(airbag_survivors_counts.index, title='Seatbelt Use', loc='upper left', fontsize=12, title_fontsize=14, bbox_to_anchor=(0.75, 1))
plt.show()

airbag_deceased_counts = df[df['dead'] == 'DEAD']['airbag'].value_counts()
plt.figure(figsize=(8, 6))
plt.pie(airbag_deceased_counts, startangle=90, colors=['#E53935', '#FF8A65'])
plt.title('Airbag Deployment Among Deceased')
plt.legend(airbag_deceased_counts.index, title='Seatbelt Use', loc='upper left', fontsize=12, title_fontsize=14, bbox_to_anchor=(0.75, 1))
plt.show()

seatbelt_survival_counts = df[df['dead'] == 'ALIVE']['seatbelt'].value_counts()
plt.figure(figsize=(8, 6))
plt.pie(seatbelt_survival_counts, startangle=90, colors=['#1E88E5', '#90CAF9'])
plt.title('Seatbelt Use Among Survivors')
plt.legend(seatbelt_survival_counts.index, title='Seatbelt Use', loc='upper left', fontsize=12, title_fontsize=14, bbox_to_anchor=(0.75, 1))
plt.show()

seatbelt_deceased_counts = df[df['dead'] == 'DEAD']['seatbelt'].value_counts()
plt.figure(figsize=(8, 6))
plt.pie(seatbelt_deceased_counts, startangle=90, colors=['#FB8C00', '#FFD54F'])
plt.title('Seatbelt Use Among Deceased')
plt.legend(seatbelt_deceased_counts.index, title='Seatbelt Use', loc='upper left', fontsize=12, title_fontsize=14, bbox_to_anchor=(0.75, 1))
plt.show()

table = pd.crosstab([df['frontal'], df['sex']], df['dead'])

plt.rcParams.update({'font.size': 24})
plt.figure(figsize=(10, 6))
mosaic(table.stack())
plt.title('Mosaic Plot: Frontal Collision, Sex, and Death')
plt.show()

print("\n" + "="*80)
print("ANALYSIS COMPLETE".center(80))
print("="*80)