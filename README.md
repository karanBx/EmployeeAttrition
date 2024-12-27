
# Employee Attrition Prediction

Understanding Workforce Turnover Through Data-Driven Insights, 
Building Predictive Models to enhance employee retention strategies.

## 

![App Screenshot](https://blogimage.vantagecircle.com/content/images/2023/11/Employee-Attrition--Meaning--Types--Reason--and-How-to-Reduce.png)



## Description


Employee attrition, or workforce turnover, is a critical metric for organizations striving to maintain stability, productivity, and profitability. High attrition rates can result in significant costs, including recruitment expenses, training new hires, and the loss of institutional knowledge.

This project explores employee attrition by analyzing various factors such as job satisfaction, salary hikes, work-life balance, and demographic details. The goal is to build a machine learning model that predicts whether an employee is likely to leave the organization. By leveraging this predictive capability, businesses can identify key drivers of attrition and implement proactive strategies to improve employee retention.
## Project Aim

The aim of this project is to analyze employee attrition data to identify the key factors influencing workforce turnover and develop a machine learning model that predicts whether an employee is likely to leave the organization. By achieving this, the project seeks to provide actionable insights that help businesses improve employee retention and make informed decisions to enhance organizational stability.
## Key Features

- Exploratory Data Analysis (EDA): Comprehensive analysis of employee attributes and attrition patterns to uncover key insights.
- Predictive Modeling: Implementation of machine learning algorithms such as K-Nearest Neighbors (KNN), Random Forest, and others to predict employee attrition.
- Feature Selection: Identification of the most influential factors affecting employee turnover.
- Model Optimization: Evaluation and tuning of multiple models to achieve the highest accuracy and reliability.
- Visualizations: Clear and insightful visualizations, including bar charts, pie charts, and heatmaps, to illustrate patterns and relationships.
- Business Insights: Actionable recommendations to help organizations address attrition and improve retention strategies.


## Tech Stack

Programming Language:   Python

Data Science Libraries: NumPy, Pandas, Scikit-learn

Visualization Libraries: Matplotlib, Seaborn, Plotly

Environment: Google Colab/Jupyter Notebook

![Python](https://img.shields.io/badge/Python-3.8-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-0.24-orange)
![Pandas](https://img.shields.io/badge/Pandas-1.3-yellow)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4-green) 











## Installation
1. Clone this repository:
   ```
   git clone https://github.com/karanBx/EmployeeAttrition.git
   ```
2. Navigate to the project directory:
   ```
   cd heart-disease-prediction
   ```
3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
    
## Workflow
1. Data Ingestion
2. Data Transformation
3. Model training with various algorithms.
4. Model evaluation and selection.
5. Deployment and usage.

## Dataset
- Source: [Analytics Employee Attrition Dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset/data?select=WA_Fn-UseC_-HR-Employee-Attrition.csv)
- Rows: 1058
- Columns: 35
- Target Variable: 'Attrition' (whether an employee left the company or not)



## Understanding Dataset

- AGE- Numerical Value
- ATTRITION- Employee leaving the company (0=no, 1=yes)
- BUSINESS TRAVEL- (1=No Travel, 2=Travel Frequently, 3=Tavel Rarely)
- DAILY RATE- Numerical Value - Salary Level
- DEPARTMENT- (1=HR, 2=R&D, 3=Sales)
- DISTANCE FROM HOME- Numerical Value - THE DISTANCE FROM WORK TO HOME

- EDUCATION- Numerical Value
- EDUCATION FIELD-	(1=HR, 2=LIFE SCIENCES, 3=MARKETING, 4=MEDICAL SCIENCES, 5=OTHERS, 6= TEHCNICAL)
- EMPLOYEE COUNT- Numerical Value
- EMPLOYEE NUMBER- Numerical Value - EMPLOYEE ID
- ENVIROMENT SATISFACTION- Numerical Value - SATISFACTION WITH THE ENVIROMENT
- GENDER- (1=FEMALE, 2=MALE)
- HOURLY RATE- Numerical Value - HOURLY SALARY
- JOB INVOLVEMENT- Numerical Value - JOB INVOLVEMENT
- JOB LEVEL- Numerical Value - LEVEL OF JOB
- JOB ROLE- (1=HC REP, 2=HR, 3=LAB TECHNICIAN, 4=MANAGER, 5= MANAGING DIRECTOR, 6= REASEARCH DIRECTOR, 7= RESEARCH SCIENTIST, 8=SALES EXECUTIEVE, 9= SALES REPRESENTATIVE)
- JOB SATISFACTION- Numerical Value - SATISFACTION WITH THE JOB
- MARITAL STATUS- (1=DIVORCED, 2=MARRIED, 3=SINGLE)
- MONTHLY INCOME- Numerical Value - MONTHLY SALARY
- MONTHY RATE- Numerical Value - MONTHY RATE
- NUMCOMPANIES WORKED- Numerical Value - NO. OF COMPANIES WORKED AT
- OVER 18- (1=YES, 2=NO)
- OVERTIME- (1=NO, 2=YES)
- PERCENT SALARY HIKE	Numerical Value - PERCENTAGE INCREASE IN SALARY.
  The parentage of change in salary between 2 year (2017, 2018).
- PERFORMANCE RATING- Numerical Value - ERFORMANCE RATING
- RELATIONS SATISFACTION- Numerical Value - RELATIONS SATISFACTION
- STANDARD HOURS- Numerical Value - STANDARD HOURS
- STOCK OPTIONS LEVEL- Numerical Value - STOCK OPTIONS.
  How much company stocks you own from this company
- TOTAL WORKING YEARS- Numerical Value - TOTAL YEARS WORKED
- TRAINING TIMES LAST YEAR- Numerical Value - HOURS SPENT TRAINING
- WORK LIFE BALANCE- Numerical Value - TIME SPENT BEWTWEEN WORK AND OUTSIDE
- YEARS AT COMPANY- Numerical Value - TOTAL NUMBER OF YEARS AT THE COMPNAY
- YEARS IN CURRENT ROLE- Numerical Value -YEARS IN CURRENT ROLE
- YEARS SINCE LAST PROMOTION- Numerical Value - LAST PROMOTION
- YEARS WITH CURRENT MANAGER- Numerical Value - YEARS SPENT WITH CURRENT MANAGER
## Analytical Interpretation

1) People are tending to switch to a different jobs at the start of their careers, or at the earlier parts of it. Once they have settled with a family or have found stability in their jobs, they tend to stay long in the same organization- only going for vertical movements in the same organization.
2) Salary and stock ptions have a great motivation on the employees and people tend to leave the organization much lesser. Higher pay and more stock options have seen more employees remain loyal to their company.
3) Work life balance is a great motivation factor for the employees. However, people with a good work-life balance, tend to switch in search of better opportunities and a better standard of living.
4) Departments where target meeting performance is very much crucial (for e.g. Sales) tend to have a greater chances of leaving the organization as compared to departments with more administration perspective (For e.g. Human Resources)
5) People with a good Job Satisfaction and Environment satisfaction are loyal to the organization- and this speaks loud for any Organization. However, people who are not much satisfied with their current project- tend to leave the organization far more.
## 

![App Screenshot](https://hrforecast.com/wp-content/uploads/2022/08/Infographic.png)

## Model Conclusion


From the results of dataset analysis and implementation of machine learning models, it can be concluded as follows:

Logistic regression was comparatively the best machine-learning model followed by Support Vector Machine (Linear Kernel) and KNN respectively. This is because this model fits well with train and test data.

The prediction results on test data, dummy data, and the complete machine learning pipeline have been successfully exported for other purposes. In addition, data exploration has also been successfully carried out using the ydata-profiling, seaborn, and matplotlib libraries.

Several improvements can be implemented in the following research/notebook. Another example is performing advanced hyperparameter tuning experiments to obtain higher accuracy (~90%).
## References

-An Introduction to Logistic Regression in Python by Simplilearn:
https://www.simplilearn.com/tutorials/machine-learning-tutorial/logistic-regression-in-python

-What Is K-Nearest Neighbor? An ML Algorithm to Classify Data by Amal Joby:
https://learn.g2.com/k-nearest-neighbor

-Decision Tree Classification Algorithm by Javatpoint
https://www.javatpoint.com/machine-learning-decision-tree-classification-algorithm

-Decision Tree vs. Random Forest – Which Algorithm Should you Use? by Abhishek Sharma
https://www.analyticsvidhya.com/blog/2020/05/decision-tree-vs-random-forest-algorithm/

-Understanding Random Forest by Tony Yiu
https://towardsdatascience.com/understanding-random-forest-58381e0602d2

-Understanding Boosting - DataSciencedojo
https://datasciencedojo.com/blog/boosting-algorithms-in-machine-learning/
## Contributions

Contributions are welcome! Please open an issue or submit a pull request for any improvements.

