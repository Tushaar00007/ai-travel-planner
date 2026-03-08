import pandas as pd
import os

path = '/Users/tushaarrohatgi/Developer/planner/Ml_model/dataset/'
files = [
    'tourism_dataset_llm_ready.csv',
    'tourism_dataset_top350_india.csv',
    'updated_pan_india_tourism_dataset.csv'
]

dataframes = []
for f in files:
    full_path = os.path.join(path, f)
    if os.path.exists(full_path):
        dataframes.append(pd.read_csv(full_path))
    else:
        print(f"Warning: {f} not found.")

if not dataframes:
    print("Error: No dataframes to merge.")
    exit(1)

# Concatenate all dataframes
merged_df = pd.concat(dataframes, ignore_index=True)

# Drop duplicate rows based on Name, City, and State to be safe
before_dedup = len(merged_df)
merged_df = merged_df.drop_duplicates(subset=['Name', 'City', 'State'], keep='first')
after_dedup = len(merged_df)
print(f"Removed {before_dedup - after_dedup} duplicate rows.")

# Remove columns which are entirely empty
# "empty" usually means NaN, but some datasets use "Unknown" or "None"
# The user said "do not hav the data basically it is empty"
# I'll drop columns where 100% of values are NaN
before_cols = len(merged_df.columns)
merged_df = merged_df.dropna(axis=1, how='all')
after_cols = len(merged_df.columns)
print(f"Dropped {before_cols - after_cols} entirely empty columns.")

# Optional: Drop columns that are >95% empty or only contain "Unknown" if needed?
# But user said "do not have the data basically it is empty" - strictly 'all' is safer unless they meant 'mostly' empty.
# Let's check for "Unknown" as well just in case.
def is_totally_empty(col):
    return col.isnull().all() or (col.astype(str).str.lower().isin(['unknown', 'none', 'nan', 'not available']).all())

cols_to_drop = [c for c in merged_df.columns if is_totally_empty(merged_df[c])]
merged_df = merged_df.drop(columns=cols_to_drop)
print(f"Dropped {len(cols_to_drop)} additional columns that only contained placeholders ('Unknown', etc.).")

# Save the new dataset
output_path = os.path.join(path, 'final_merged_tourism_dataset.csv')
merged_df.to_csv(output_path, index=False)
print(f"New dataset created at: {output_path}")
print(f"Final shape: {merged_df.shape}")
