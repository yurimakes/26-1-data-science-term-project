import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# df = pd.read_csv('bank_direct_marketing_campaigns.csv')
# # data size
# print(df)


# # Data Overview
# print(df.info())


# # Data Integrity
# # Identify the total number of duplicate rows
# # The duplicated() function returns True for each row that is a duplicate.
# # sum() calculates the total count of these True values.
# duplicate_count = df.duplicated().sum()

# print(f"--- Data Integrity Check Results ---")
# print(f"Total number of records: {len(df)}")
# print(f"Number of duplicate records: {duplicate_count}")

# # Inspect duplicate data if any exist
# if duplicate_count > 0:
#     print("\n--- Duplicate Data Samples (First 5 rows) ---")

#     # Setting keep=False displays all instances of the duplicates
#     duplicates = df[df.duplicated(keep=False)]
#     print(duplicates.sort_values(by=list(df.columns)).head())
        
# else:
#     print("\nResult: No duplicate records found in the dataset.")
#     print("Data integrity is confirmed to be maintained.")



# # Target Class Distribution
# # Calculate frequencies and percentages
# target_counts = df['y'].value_counts()
# target_percent = df['y'].value_counts(normalize=True) * 100

# print("--- Target Variable (y) Distribution ---")
# print("Counts:")
# print(target_counts)
# print("\nPercentages (%):")
# print(target_percent)

# # Visualization
# plt.figure(figsize=(8, 6))
# sns.countplot(x='y', data=df, palette='viridis')
# plt.title('Distribution of Subscription (Target Variable y)')
# plt.xlabel('Subscribed (y)')
# plt.ylabel('Number of Clients')

# for i, count in enumerate(target_counts):
#     plt.text(i, count + 500, f'{target_percent.iloc[i]:.1f}%', ha='center', fontsize=12)

# plt.tight_layout()
# plt.show()



# # Removing Duplicate
# # Record the initial number of rows
# initial_rows = len(df)
# print(f"Initial number of records: {initial_rows}")

# # Remove duplicate records
# # drop_duplicates() removes rows where all column values are identical.
# # inplace=True applies the change directly to the existing dataframe.
# df.drop_duplicates(inplace=True)

# # Record the number of rows after cleaning
# final_rows = len(df)
# removed_rows = initial_rows - final_rows

# print(f"Number of records after removing duplicates: {final_rows}")
# print(f"Total number of duplicates removed: {removed_rows}")

# # Verification
# if removed_rows > 0:
#     print("\nSuccess: Duplicates have been successfully removed.")
#     print(f"The dataset now contains {final_rows} unique observations.")
# else:
#     print("\nNo duplicates were found or they were already removed.")

# # Save the cleaned dataset
# df.to_csv('duplicate_removed.csv', index=False)



# # Handling 'unknown' Values
# df = pd.read_csv('duplicate_removed.csv')
# df.replace('unknown', np.nan, inplace=True)

# print("--- Missing Values (NaN) count after replacing 'unknown' ---")
# null_counts = df.isnull().sum()
# print(null_counts[null_counts > 0])
# print(f"Total missing values: {null_counts.sum()}\n")

# categorical_cols = df.select_dtypes(include=['object', 'string']).columns

# for col in categorical_cols:
#     if df[col].isnull().sum() > 0:
#         # Calculate the most frequent value 
#         mode_val = df[col].mode()[0]

#         # Fill the NaNs with the mode
#         df[col] = df[col].fillna(mode_val)
#         print(f"Imputed [{col}] with mode: '{mode_val}'")

# # Final verification to ensure no nulls remain
# final_verification = df.isnull().sum().sum()
# print(f"\n--- Final Verification ---")
# print(f"Remaining null values: {final_verification}")

# if final_verification == 0:
#     print("Success: All 'unknown' values have been successfully handled.")

#     # Save the final preprocessed dataset for encoding
#     df.to_csv('data_cleaning_finished.csv', index=False)



# # Categorical Encoding
# # Label encoding for binary features 
# # We map 'yes' to 1 and 'no' to 0 for binary categorical variables.
# df = pd.read_csv('data_cleaning_finished.csv')
# binary_cols = ['y', 'default', 'housing', 'loan']
# for col in binary_cols:
#     df[col] = df[col].map({'yes': 1, 'no': 0})
#     print(f"Encoded binary column [{col}] (yes:1, no:0)")

# # One-Hot encoding for nominal features
# nominal_cols = ['job', 'marital', 'education', 'contact', 'month', 'day_of_week', 'poutcome']
# df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)

# print("\n--- Data Transformation (Encoding) Finished ---")
# print(f"New shape of the dataset: {df.shape}")
# print(df.head())

# # Save the encoded dataset for scaling
# df.to_csv('data_encoded.csv', index=False)



# Numerical Feature Scaling
df = pd.read_csv('data/processed/data_encoded.csv')
print(f"Successfully loaded 'data_encoded.csv'. Shape: {df.shape}")

# Define numerical features to be scaled
numerical_features = [
    'age', 'campaign', 'pdays', 'previous',
    'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
    'euribor3m', 'nr.employed'
]

# Z-score normalization
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

print("\n--- Scaling Complete ---")

# Check mean and standard deviation
scaling_stats = df[numerical_features].describe().loc[['mean', 'std']]
print("\nVerification Statistics:")
print(scaling_stats)

df.to_csv('data/processed/data_final_preprocessed.csv', index=False)
print('\nFinal dataset saved')