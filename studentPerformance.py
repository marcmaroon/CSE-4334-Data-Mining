import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Set publication-quality plot style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})

# 1. LOAD DATA (The CSVs in this dataset use semicolon ';' as a delimiter)
df_mat = pd.read_csv("student-mat.csv", sep=";")


df = df_mat.copy() #using student-mat.csv for analysis

# 2. Summary Statistics Table (Mean, Median, Std Dev)
numeric_cols = ['age', 'absences', 'G1', 'G2', 'G3']
summary_stats = df[numeric_cols].agg(['mean', 'median', 'std', 'min', 'max']).T
print("=== SUMMARY STATISTICS ===")
print(summary_stats)

# Save summary stats to CSV/Excel for your presentation slides table
summary_stats.to_csv("summary_stats_table.csv")


# VISUAL 1: Histogram of Final Grade (G3) with Mean & Median lines
plt.figure(figsize=(8, 5))
mean_g3 = df['G3'].mean()
median_g3 = df['G3'].median()
std_g3 = df['G3'].std()

sns.histplot(df['G3'], kde=True, color='skyblue', bins=20)
plt.axvline(mean_g3, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_g3:.2f}')
plt.axvline(median_g3, color='green', linestyle='-', linewidth=2, label=f'Median: {median_g3:.2f}')

plt.title(f'Distribution of Final Grades (G3)\nStd Dev = {std_g3:.2f}')
plt.xlabel('Final Grade (0-20 scale)')
plt.ylabel('Student Count')
plt.legend()
plt.tight_layout()
plt.savefig('fig1_grade_distribution.png', dpi=300)
plt.show()


# VISUAL 2: Box Plot - Final Grade vs. Weekly Study Time-------
# Mapping studytime numeric values to descriptive labels
study_map = {1: '<2 hrs', 2: '2-5 hrs', 3: '5-10 hrs', 4: '>10 hrs'}
df['studytime_label'] = df['studytime'].map(study_map)

plt.figure(figsize=(8, 5))
sns.boxplot(x='studytime_label', y='G3', data=df, 
            order=['<2 hrs', '2-5 hrs', '5-10 hrs', '>10 hrs'], 
            palette='Blues', hue = 'studytime_label', legend=False)

plt.title('Final Grade Distribution by Weekly Study Time')
plt.xlabel('Weekly Study Time')
plt.ylabel('Final Grade (G3)')
plt.tight_layout()
plt.savefig('fig2_grade_by_studytime_boxplot.png', dpi=300)
plt.show()

#----Pre processing SECTION---------
#Separate features (X) from target y, only concerned with G3 (y)
X = df.drop(columns=['G3']) # drops
y = df['G3'] #G3 trying to predict it 

print("\n=== FEATURES (X) ===")
print(X.shape)
print("\n=== TARGET (y) ===")
print(y.shape)
#remove label study label since i dont want giving it to model
X = X.drop(columns=['studytime_label'])

#Identify categorical and numerical features // dont think i need this but just to make sure im having the right features
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(exclude=['object']).columns
print("\n=== CATEGORICAL FEATURES ===")
print(list(categorical_cols)) # may need to revisit 
print("\n=== NUMERICAL FEATURES ===")
print(list(numerical_cols)) # since i dropped G3 from the colum it wont show ig


#Encode categorical variables-----------
#example school GP or MS should be 1 or 0
preprocessor = ColumnTransformer(
    transformers=[
        ('categorical', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough' #let the rest numerical values the same 
)

