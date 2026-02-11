
import pandas as pd

# Load the dataset
df = pd.read_csv('data/dataset-MVP-Mar2025.csv')

# Clean column names
def clean_col_names(df):
    cols = df.columns
    new_cols = []
    for col in cols:
        new_col = col.strip()
        new_col = new_col.replace(' ', '_')
        new_col = new_col.replace('(', '')
        new_col = new_col.replace(')', '')
        new_col = new_col.replace('-', '_')
        new_col = new_col.replace('/', '_')
        new_col = new_col.replace('.', '')
        new_col = new_col.lower()
        new_cols.append(new_col)
    df.columns = new_cols
    return df

df = clean_col_names(df)

# Remove rows with all NaN values
df.dropna(how='all', inplace=True)

# Save the cleaned data
df.to_csv('data/cleaned_dataset.csv', index=False)

print("Data cleaning and saving complete.")
