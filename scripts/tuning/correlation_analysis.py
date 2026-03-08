import matplotlib
matplotlib.use('Agg')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the processed dataset
try:
    df = pd.read_csv('/workspaces/Buying-vs.-Renting-a-Property-in-Toronto/data/processed_data.csv')
except FileNotFoundError:
    print("Error: 'data/processed_dataset.csv' not found.")
    exit()

df_numeric = df.select_dtypes(include=['number'])


# Calculate the correlation matrix
correlation_matrix = df_numeric.corr()

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of All Features')
plt.savefig('correlation_heatmap.png')
plt.close()

print("Correlation heatmap saved to correlation_heatmap.png")

# Optional: To see the correlations with the target variable "Log_Return_MoM" more clearly
print("\nCorrelations with Log_Return_MoM:")
print(correlation_matrix['Log_Return_MoM'].sort_values(ascending=False))
