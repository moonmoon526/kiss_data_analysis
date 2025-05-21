# === Part 1: Load data and preprocess ===
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.lines import Line2D
import numpy as np
import os

# Load Excel data file with error handling
file_path = r"C:\Users\USER\Downloads\DATA_Kiss_count_gender_and_IQ.xlsx"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file {file_path} does not exist. Please check the path.")

try:
    df = pd.read_excel(file_path)
except Exception as e:
    raise Exception(f"Error loading Excel file: {e}")

# Print column names and sample data for debugging
print("Column names:", df.columns.tolist())
print("\nFirst 5 rows of the DataFrame:\n", df.head())
print("\nData types:\n", df.dtypes)

# Drop unnamed columns (if any)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Data validation
required_columns = ['Name', 'Gender', 'IQ', 'Kiss Count', 'Age of First Kiss']
if not all(col in df.columns for col in required_columns):
    raise ValueError(f"Missing required columns. Expected: {required_columns}")

# Filter valid data rows (exclude statistical tables)
df = df[df['Name'].apply(lambda x: isinstance(x, str) and not x.lower().startswith(('gender', 'iq', 'kiss count', 'age of first kiss')))]
print("\nRows after filtering:", len(df))
print("Unique Gender values (raw):", df['Gender'].unique())

# Create Gender_binary from Gender column
df['Gender'] = df['Gender'].astype(str).str.lower().str.strip()
df['Gender_binary'] = df['Gender'].map({'male': 1, 'female': 0})

# Check for invalid or missing gender values
if df['Gender_binary'].isna().any():
    invalid_rows = df[df['Gender_binary'].isna()][['Name', 'Gender']]
    print(f"Invalid or missing gender values found in rows:\n{invalid_rows}")
    df = df.dropna(subset=['Gender_binary'])
    print(f"Dropped rows with invalid gender values. Remaining rows: {len(df)}")

if df.empty:
    raise ValueError("No valid data remains after processing gender values.")

# Ensure all necessary columns are numeric
numeric_cols = ['IQ', 'Kiss Count', 'Age of First Kiss', 'Gender_binary']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Drop rows with missing values in numeric columns
df = df.dropna(subset=numeric_cols)
if df.empty:
    raise ValueError("No valid data remains after dropping missing values.")

n = len(df)
print(f"Final number of rows: {n}")

# Compare with Excel correlation table
excel_correlations = {
    'Gender vs IQ': 0.1282132684,
    'Gender vs Kiss Count': -0.0034629037,
    'Gender vs Age of First Kiss': -0.1940255144,
    'IQ vs Kiss Count': 0.1108121116,
    'IQ vs Age of First Kiss': -0.0418415307,
    'Kiss Count vs Age of First Kiss': -0.2307638949
}

# === Part 2: Gender vs Kiss Count ===
r_pb, p_pb = stats.pointbiserialr(df['Gender_binary'], df['Kiss Count'])
t_pb = r_pb * np.sqrt(n - 2) / np.sqrt(1 - r_pb**2)
p_pb_t = 2 * (1 - stats.t.cdf(abs(t_pb), df=n-2))
print(f"\nGender vs Kiss Count: r = {r_pb:.6f}, t = {t_pb:.6f}, p (from pointbiserialr) = {p_pb:.6f}, p (from t) = {p_pb_t:.6f}")
print(f"Excel r (Gender vs Kiss Count): {excel_correlations['Gender vs Kiss Count']:.6f}")

plt.figure(figsize=(8, 5))
sns.boxplot(x='Gender', y='Kiss Count', data=df, palette={'female': 'pink', 'male': 'lightskyblue'})
plt.title(f"Kiss Count by Gender\nPoint-Biserial r = {r_pb:.4f}, t = {t_pb:.4f}, p = {p_pb:.4f}")
plt.xlabel("Gender")
plt.ylabel("Kiss Count")
plt.tight_layout()
plt.show()

# === Part 3: IQ vs Kiss Count ===
r_iq, p_iq = stats.pearsonr(df['IQ'], df['Kiss Count'])
t_iq = r_iq * np.sqrt(n - 2) / np.sqrt(1 - r_iq**2)
p_iq_t = 2 * (1 - stats.t.cdf(abs(t_iq), df=n-2))
print(f"\nIQ vs Kiss Count: r = {r_iq:.6f}, t = {t_iq:.6f}, p (from pearsonr) = {p_iq:.6f}, p (from t) = {p_iq_t:.6f}")
print(f"Excel r (IQ vs Kiss Count): {excel_correlations['IQ vs Kiss Count']:.6f}")

plt.figure(figsize=(9, 6))
sns.regplot(x='IQ', y='Kiss Count', data=df, ci=95, scatter_kws={"s": 70}, color='gray')
plt.title(f"IQ vs Kiss Count\nPearson r = {r_iq:.4f}, t = {t_iq:.4f}, p = {p_iq:.4f}")
plt.xlabel("IQ")
plt.ylabel("Kiss Count")
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# === Part 4: Age of First Kiss vs IQ and Kiss Count ===
r_afk_iq, p_afk_iq = stats.pearsonr(df['Age of First Kiss'], df['IQ'])
t_afk_iq = r_afk_iq * np.sqrt(n - 2) / np.sqrt(1 - r_afk_iq**2)
p_afk_iq_t = 2 * (1 - stats.t.cdf(abs(t_afk_iq), df=n-2))
print(f"\nAge of First Kiss vs IQ: r = {r_afk_iq:.6f}, t = {t_afk_iq:.6f}, p (from pearsonr) = {p_afk_iq:.6f}, p (from t) = {p_afk_iq_t:.6f}")
print(f"Excel r (IQ vs Age of First Kiss): {excel_correlations['IQ vs Age of First Kiss']:.6f}")

plt.figure(figsize=(9, 5))
sns.regplot(x='Age of First Kiss', y='IQ', data=df, ci=95, scatter_kws={"s": 70}, color='teal')
plt.title(f"Age of First Kiss vs IQ\nr = {r_afk_iq:.4f}, t = {t_afk_iq:.4f}, p = {p_afk_iq:.4f}")
plt.xlabel("Age of First Kiss")
plt.ylabel("IQ")
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

r_afk_kiss, p_afk_kiss = stats.pearsonr(df['Age of First Kiss'], df['Kiss Count'])
t_afk_kiss = r_afk_kiss * np.sqrt(n - 2) / np.sqrt(1 - r_afk_kiss**2)
p_afk_kiss_t = 2 * (1 - stats.t.cdf(abs(t_afk_kiss), df=n-2))
print(f"\nAge of First Kiss vs Kiss Count: r = {r_afk_kiss:.6f}, t = {t_afk_kiss:.6f}, p (from pearsonr) = {p_afk_kiss:.6f}, p (from t) = {p_afk_kiss_t:.6f}")
print(f"Excel r (Kiss Count vs Age of First Kiss): {excel_correlations['Kiss Count vs Age of First Kiss']:.6f}")

plt.figure(figsize=(9, 5))
sns.regplot(x='Age of First Kiss', y='Kiss Count', data=df, ci=95, scatter_kws={"s": 70}, color='indianred')
plt.title(f"Age of First Kiss vs Kiss Count\nr = {r_afk_kiss:.4f}, t = {t_afk_kiss:.4f}, p = {p_afk_kiss:.4f}")
plt.xlabel("Age of First Kiss")
plt.ylabel("Kiss Count")
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# === Part 5: Boxplots comparing Gender differences ===
plt.figure(figsize=(16, 5))

plt.subplot(1, 3, 1)
sns.boxplot(x='Gender', y='IQ', data=df, palette={'female': 'pink', 'male': 'lightskyblue'})
plt.title("IQ by Gender")

plt.subplot(1, 3, 2)
sns.boxplot(x='Gender', y='Age of First Kiss', data=df, palette={'female': 'pink', 'male': 'lightskyblue'})
plt.title("Age of First Kiss by Gender")

plt.subplot(1, 3, 3)
sns.boxplot(x='Gender', y='Kiss Count', data=df, palette={'female': 'pink', 'male': 'lightskyblue'})
plt.title("Kiss Count by Gender")

plt.tight_layout()
plt.show()

# === Part 6: Bubble chart ===
color_map = {'male': 'lightskyblue', 'female': 'pink'}
colors = df['Gender'].map(color_map)
bubble_sizes = df['Kiss Count']**2 * 10

plt.figure(figsize=(10, 7))
plt.scatter(df['IQ'], df['Age of First Kiss'], s=bubble_sizes, c=colors, alpha=0.6, edgecolors='black')

try:
    from adjustText import adjust_text
    texts = [plt.text(row['IQ'], row['Age of First Kiss'], row['Name'], fontsize=8, ha='center', va='center') for _, row in df.iterrows()]
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
except ImportError:
    for i, row in df.iterrows():
        plt.text(row['IQ'], row['Age of First Kiss'], row['Name'], fontsize=8, ha='center', va='center')

plt.xlabel("IQ")
plt.ylabel("Age of First Kiss")
plt.title("Bubble Chart: IQ vs Age of First Kiss\n(Bubble = Kiss Count, Color = Gender)")
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(handles=[
    Line2D([0], [0], marker='o', color='w', label='Male', markerfacecolor='lightskyblue', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Female', markerfacecolor='pink', markersize=10)
], title='Gender')
plt.tight_layout()
plt.show()