import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

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

# Save summary stats to CSV/Excel for presentation slides table
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
X = df.drop(columns=['G3'])
y = df['G3'] #G3 trying to predict it 
X = X.drop(columns=['studytime_label']) #remove label study label since i dont want giving it to model

print("\n=== FEATURES (X) ===")
print(X.shape)
print("\n=== TARGET (y) ===")
print(y.shape)


#listing categorical and numerical features / nice to see the features
categorical_cols = X.select_dtypes(include=['object', 'string']).columns
numerical_cols = X.select_dtypes(exclude=['object']).columns
print("\n=== CATEGORICAL FEATURES ===")
print(list(categorical_cols))
print("\n=== NUMERICAL FEATURES ===")
print(list(numerical_cols)) 


#Encode categorical variables-----------
#example school GP or MS should be 1 or 0
preprocessor = ColumnTransformer(
    transformers=[
        ('categorical', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough' #let the rest numerical values the same 
)
#splitting data set  80 percent training and 20 percent testing -----  part of Step 3---------
X_train, X_test, y_train, y_test = train_test_split( # 79 testing and 316 training samples 80/20 split
    X,
    y,
    test_size=0.20,
    random_state=42 # to be able to make same split again everytime Program runs
)

print("\n=== TRAINING AND TESTING DATA ===")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

#applying the preprocessing 
X_train_processed = preprocessor.fit_transform(X_train) # Fit calc mean, std, etc. and transform training data
X_test_processed = preprocessor.transform(X_test) #transform test data using the same parameters learned from training data

print("\n=== PROCESSED DATA ===")
print("Training shape:", X_train_processed.shape) 
print("Testing shape:", X_test_processed.shape)

#3 Model Training and Performance  Evaluation---------------------------
model = RandomForestRegressor(
    n_estimators=100, #100 decision trees in forest
    random_state=42
)
#student info -> train-> encode data-> learn relationships between features and G3 -> predict G3 for new students
model.fit(X_train_processed, y_train)

print("\n=== MODEL TRAINING COMPLETE ===")

#now making predictions
y_pred = model.predict(X_test_processed)

print("\n=== PREDICTIONS ===")
print("Actual grades:   ", y_test.values[:10])
print("Predicted grades:", y_pred[:10])

#Calculating MSE and R^2 
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n=== MODEL PERFORMANCE ===")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R² Score: {r2:.4f}")

#Model Performance Visualization: Actual vs. Predicted Grades
plt.figure(figsize=(8, 5))

plt.scatter(y_test, y_pred)

# Perfect prediction line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle='--'
)

plt.title('Actual vs. Predicted Final Grades')
plt.xlabel('Actual Final Grade (G3)')
plt.ylabel('Predicted Final Grade (G3)')

plt.tight_layout()
plt.savefig('fig3_actual_vs_predicted.png', dpi=300)
plt.show()

#getting importance features
feature_names = preprocessor.get_feature_names_out()

importance = model.feature_importances_

feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
})

feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)

print("\n=== TOP 10 MOST IMPORTANT FEATURES ===")
print(feature_importance.head(10))
