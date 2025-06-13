# Student Habits Analysis Project

This repository contains a full analysis of how student habits impact academic performance, specifically exam scores.

The goal of this project was to explore:

- How much study hours, attendance, extracurricular participation, and parental education influence student outcomes.
- How well these variables can be used to predict performance via a multiple linear regression model.

---

## 🔍 Summary of Results

- **Study Hours per Day** → Very strong positive predictor of exam score  
- **Attendance Percentage** → Weak but significant positive effect  
- **Parental Education Level** → No significant predictive power detected  
- **Extracurricular Participation** → No significant impact  

The final regression model achieved an R² of ~0.68, meaning ~68% of the variance in exam scores was explained by the selected predictors.

---

## 📄 Reports

- 👉 [Habits Analysis Report (PDF)](Student_Habits_Analysis_Report.pdf)
- 👉 [Regression Report (PDF)](Student_Habits_Regression_Report.pdf)

---

## ⚙️ Methodology

- Data Cleaning: pandas, mapping categorical features
- Visualization: matplotlib, seaborn
- Statistical Tests: Pearson & Spearman correlations, ANOVA
- Regression Modeling: scikit-learn Linear Regression, statsmodels OLS
- Report Generation: matplotlib, Python-docx, PDF output

---

## 📂 Files

| File | Description |
|------|-------------|
| `Student_Habits_Analysis.py` | Main analysis code |
| `Student_Habits_Regression_Report.pdf` | Regression model report |
| `Student_Habits_Combined_Report.pdf` | Combined analysis report |
| `cleaned_student_performance.csv` | Cleaned dataset used for analysis |

---

## 💻 How to Run

1. Clone this repo
2. Run `Student_Habits_Analysis.py` in your Python environment  
    Required packages: `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `statsmodels`, `docx`, `PyPDF2`

---

## 📢 About

This project was created as part of my data analytics portfolio to showcase:

✅ Data analysis & visualization skills  
✅ Model building & statistical testing  
✅ Ability to communicate insights clearly  

---

*Author: Trevor Smith*  
GitHub: [https://github.com/trevorsmith00](https://github.com/trevorsmith00)  
