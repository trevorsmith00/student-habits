import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway, spearmanr, pearsonr

# Get current working directory
current_dir = os.getcwd()

# Use relative path
file_path = os.path.join(current_dir, 'student_habits_performance.csv')

# Load CSV
df = pd.read_csv(file_path)


# Display the first few rows and basic info for an overview
df_info = df.info()
df_head = df.head()

df_info, df_head

# Drop irrelevant column
df_cleaned = df.drop(columns=['student_id'])

# Convert categorical variables to numeric using mapping or one-hot encoding where appropriate
# Binary encodings
df_cleaned['gender'] = df_cleaned['gender'].map({'Male': 0, 'Female': 1})
df_cleaned['part_time_job'] = df_cleaned['part_time_job'].map({'No': 0, 'Yes': 1})
df_cleaned['extracurricular_participation'] = df_cleaned['extracurricular_participation'].map({'No': 0, 'Yes': 1})

# Ordinal encodings for diet quality and internet quality
diet_mapping = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Excellent': 3}
internet_mapping = {'Poor': 0, 'Average': 1, 'Good': 2, 'Excellent': 3}
education_mapping = {
    'None': 0, 'High School': 1, 'Associate': 2,
    'Bachelor': 3, 'Master': 4, 'Doctorate': 5
}

df_cleaned['diet_quality'] = df_cleaned['diet_quality'].map(diet_mapping)
df_cleaned['internet_quality'] = df_cleaned['internet_quality'].map(internet_mapping)
df_cleaned['parental_education_level'] = df_cleaned['parental_education_level'].map(education_mapping)

print(df_cleaned.head())
df_cleaned.to_csv("cleaned_student_performance.csv", index=False)

df_cleaned.info()

# Create a DataFrame with attendance, study hours, and exam scores
features = ['attendance_percentage', 'study_hours_per_day', 'exam_score']
df_attendance_study = df_cleaned[features].dropna()

# Plot 1: Attendance vs Exam Score
plt.figure(figsize=(8, 6))
sns.set_theme(style="whitegrid", font_scale=1.2, palette="muted")
sns.regplot(data=df_attendance_study, x='attendance_percentage', y='exam_score', scatter_kws={'alpha':0.5,'s':40}, color='blue')
plt.title("Impact of Attendance on Exam Performance", fontsize=16, weight='bold')
plt.xlabel("Attendance (%)", fontsize=12)
plt.ylabel("Exam Score", fontsize=12)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.show()

attendance = df_cleaned['attendance_percentage']
exam_scores = df_cleaned['exam_score']

pearson_corr, pearson_p = pearsonr(attendance, exam_scores)
spearman_corr, spearman_p = spearmanr(attendance, exam_scores)

print({
    "Pearson Correlation Coefficient": pearson_corr,
    "Pearson p-value": pearson_p,
    "Spearman Correlation Coefficient": spearman_corr,
    "Spearman p-value": spearman_p
})

# Plot 2: Study Hours vs Exam Score
plt.figure(figsize=(8, 6))
sns.set_theme(style="whitegrid", font_scale=1.2, palette="muted")
sns.regplot(data=df_attendance_study, x='study_hours_per_day', y='exam_score', scatter_kws={'alpha':0.4}, color='green')
plt.title("Impact of Study Hours on Exam Performance", fontsize=16, weight='bold')
plt.xlabel("Study Hours / Day")
plt.ylabel("Exam Score")
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.show()

# Extract study hours and exam score columns
study_hours = df_cleaned['study_hours_per_day']
exam_scores = df_cleaned['exam_score']

# Perform Pearson and Spearman correlation tests
pearson_corr, pearson_p = pearsonr(study_hours, exam_scores)
spearman_corr, spearman_p = spearmanr(study_hours, exam_scores)

print({
    "Pearson Correlation Coefficient": pearson_corr,
    "Pearson p-value": pearson_p,
    "Spearman Correlation Coefficient": spearman_corr,
    "Spearman p-value": spearman_p
})


# Calculate correlation coefficients
attendance_corr = df_attendance_study['attendance_percentage'].corr(df_attendance_study['exam_score'])
study_corr = df_attendance_study['study_hours_per_day'].corr(df_attendance_study['exam_score'])

attendance_corr, study_corr

# Re-map education levels for labeling
education_labels = {
    0: "None",
    1: "High School",
    2: "Associate",
    3: "Bachelor",
    4: "Master",
    5: "Doctorate"
}

# Prepare the dataset for visualization
df_viz = df_cleaned[['parental_education_level', 'exam_score']].dropna()
df_viz['parental_education_label'] = df_viz['parental_education_level'].map(education_labels)
order = ["None", "High School", "Associate", "Bachelor", "Master", "Doctorate"]

# Set theme and font
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)
plt.rcParams['font.family'] = 'DejaVu Sans'


# Create the visualization
plt.figure(figsize=(10, 6))

palette = {
    "None": "#8dd3c7",
    "High School": "#bbbb53",
    "Associate": "#6da356",
    "Bachelor": "#c55043",
    "Master": "#2f73a4",
    "Doctorate": "#fdb462"
}

sns.stripplot(
    data=df_viz,
    x='parental_education_label',
    y='exam_score',
    order=order,
    jitter=0.25,
    size=5,
    alpha=0.5,
    palette=palette,
)
sns.boxplot(
    data=df_viz,
    x='parental_education_label',
    y='exam_score',
    order=order,
    showcaps=True,
    boxprops={'facecolor': 'none', 'edgecolor': 'gray'},
    whiskerprops={'color': 'gray'},
    flierprops={'marker': 'o', 'markersize': 4},
    medianprops={'color': 'firebrick'}
)


plt.title("Relationship Between Parental Education Level and Student Exam Scores", fontsize=16, weight='bold')
plt.xlabel("Parental Education Level", fontsize=12)
plt.ylabel("Exam Score", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.tight_layout()
plt.show()

# Prepare groups for ANOVA: list of exam scores grouped by parental education level
education_groups = df_viz.groupby('parental_education_label')['exam_score'].apply(list)

# Perform one-way ANOVA
anova_result = f_oneway(*education_groups)

print(anova_result.statistic, anova_result.pvalue)
