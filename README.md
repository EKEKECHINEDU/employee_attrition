## Employee Attrition Analysis  
![image title](https://img.shields.io/badge/pandas-2.1.1-red.svg) ![image title](https://img.shields.io/badge/numpy-2.1.2-green.svg) ![Image title](https://img.shields.io/badge/sklearn-1.1.1-orange.svg) ![image title](https://img.shields.io/badge/seaborn-v0.10.0-yellow.svg) ![image title](https://img.shields.io/badge/matplotlib-v3.09.1-orange.svg)

**The code is available [here](#-githubcode ipynb)**

<br>
<br>
<p align="center">
  <img src="https://github.com/EKEKECHINEDU/employee_attrition/blob/main/images/attrition-rate.jpg" 
       width="350">
</p>
<br>

<p align="center">
  <a href="#introduction"> Introduction </a> •
  <a href="#problemdef"> Problem Definition </a> •
  <a href="#goal"> Goals </a> •
  <a href="#variable"> Variable Definition </a> •
  <a href="#preprocessing"> Data Cleaning & Preprocessing </a> •
  <a href="#descriptive"> Descriptive Statistics </a> •
  <a href="#visualization"> Visualization </a> •
  <a href="#feature"> Feature Engineering</a> •
  <a href="#models"> Model Selection & Training</a> •
  <a href="#results"> Model Results</a> •
  <a href="#conclusion"> Conclusion</a> 
</p>

<a id = 'introduction'></a>
## Introduction

Employee attrition—primarily through resignations and dismissals—poses a significant challenge to organizational sustainability, making it a key focus for human resources and management. This project uncover attrition patterns from a financial services company. 


<a id = 'problemdef'></a>
## Problem Definition

High turnover rates lead to increased recruitment costs, loss of experienced employees, and potential disruptions in service delivery. Traditional methods of analysing attrition often fail to capture complex patterns and predictive indicators, limiting the ability of organizations to implement effective retention strategies

<a id = 'goal'></a>
## Goals

- Identify key predictors of attrition
- Compare the strength of classification algorithms
- Make predictions based on the chosen model

<a id = 'variable'></a>
## Variable Definition

The outcome variable is binary, equalling 1 if employee stayed and equals 0 otherwise. 
The feature variables can be grouped into the following two broad categories:
#### Demographic and Personal Attributes  
Gender; Marital Status; Age in YY.; Experience (YY.MM); Tenure   

#### Organizational and Job-Related Factors
Location; Employment Group; Function; Hiring Source; Promoted/Not Promoted; Job Role Match

<a id = 'preprocessing'></a>
## Data Cleaning and Preprocessing

Personal Identifier Information such as personal ID and telephone numbers were dropped to preserve anonymity. The dataset contained only 4 missing values in years of experience and 2 missing values in job roles, resulting in a clean dataset with no missing values

<a id = 'descriptive'></a>
## Descriptive Statistics

The low average tenure compared to significantly higher overall work experience suggests a need to explore factors influencing employee retention or turnover. Additionally, the age and experience distribution—spanning early to mid-career professionals—along with outliers, may reveal deeper insights into engagement, productivity, or training needs when segmented further.

| Variable             | Mean  | Std  | Min   | Q1    | Median | Q3    | Max   |
|----------------------|-------|------|-------|-------|--------|-------|-------|
| Tenure               | 1.21  | 0.82 | 0.00  | 0.11  | 1.06   | 2.04  | 3.00  |
| Experience (YY.MM)   | 5.16  | 3.48 | 0.03  | 2.10  | 4.11   | 7.03  | 25.08 |
| Age in YY.           | 29.08 | 4.50 | 21.05 | 26.05 | 28.06  | 31.07 | 52.06 |


<a id ='visualization'></a>
## Visualization

Code for showing combination of feature variables on outcome variable is provided as follows: 

```
# List of continuous variables
continuous_vars = ['tenure', 'experienceyymm', 'ageinyy']

# Plot density plots
#Categorical variables
categorical_vars = ['location', 'empgroup', 'function', 'gender', 'maritalstatus', 
                    'hiringsource', 'promoted', 'jobrolematch']


# Plot count plots
plt.figure(figsize=(12, 12))
for i, var in enumerate(categorical_vars, 1):
    plt.subplot(3, 3, i)
    sns.countplot(data=attrdata, x=var, hue="stayleft", palette="Set2")
    plt.xticks(rotation=60)
    plt.title(f"Distribution of {var} by Stay/Left")
    plt.xlabel(var)
    plt.ylabel("Count")

plt.tight_layout()
plt.show()
```

<br>
<p align="center">
  <img src="https://github.com/EKEKECHINEDU/employee_attrition/blob/main/images/barattrition.png">
</p>
<br>


#### Employee Characteristics
The workforce is relatively young, with an average age of 29 and tenure just over one year. While overall experience varies widely—from newcomers to highly experienced staff—differences in retention across gender and marital status highlight the need for more tailored HR policies.

#### Work Characteristics
Turnover rates vary by location, employment group, and job function, pointing to the influence of local and structural factors. Hiring source, promotion opportunities, and role alignment significantly impact retention, stressing the importance of strategic onboarding and career development.


<a id = 'feature'></a>
## Feature Engineering

**One-Hot Encoding:**  
The variables *Gender*, *Tenure Group*, *Function*, *Hiring Source*, *Marital Status*, *Employment Group*, *Job Role Match*, and *Promotion* were one-hot encoded. This technique transforms categorical variables into binary columns to avoid implying ordinal relationships and ensures that machine learning models can accurately interpret the data. It enhances model performance and interpretability by treating each category as a separate, unbiased input.

**Categorical Encoding:**  
The *Location* variable was categorically encoded by assigning unique numerical values to each category. This method preserves the nominal nature of the data while reducing dimensionality, making it especially useful for variables with many categories. It allows the model to incorporate location-based insights efficiently without adding excessive complexity.


<a id = 'models'></a>
## Model Selection & Training

The dataset was split into 80% training data and 20% test data to ensure robust evaluation

**Logistic Regression**  
Logistic Regression is a binary classification model that predicts the likelihood of an employee staying or leaving using the logistic (sigmoid) function. It provides interpretable probabilities and estimates the impact of factors like age, gender, or tenure on attrition.

**Decision Tree**  
A Decision Tree splits data into subsets based on feature values, creating a tree-like structure where each node represents a decision rule. It helps identify which features contribute most to attrition, offering clear, visual insights into employee turnover patterns.

**Random Forest**  
Random Forest is an ensemble method that combines multiple decision trees to improve prediction accuracy and reduce overfitting. It is effective with complex, high-dimensional data and highlights important features (e.g., promotions, tenure) influencing attrition.

**Naive Bayes**  
Naive Bayes is a probabilistic model based on Bayes’ Theorem that assumes feature independence. It is efficient for large datasets with many categorical features and performs well even when the independence assumption is not strictly met.


<a id = 'results'></a>
## Model Results

```
# Initialize models
lr = LogisticRegression(C=0.1, random_state=42, solver='liblinear')
dt = DecisionTreeClassifier()
rm = RandomForestClassifier()
gnb = GaussianNB()

# Prepare an empty list to store the results
results = []

# Loop through models and compute accuracies
for model, model_name in zip([lr, dt, rm, gnb], ["Logistic Regression", "Decision Tree", "Random Forest", "Naive Bayes"]):
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Calculate accuracies
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    
    # Append results to the list
    results.append([model_name, train_accuracy, test_accuracy])

# Create a DataFrame to display the results as a table
results_df = pd.DataFrame(results, columns=["Model", "Training Data Accuracy", "Test Data Accuracy"])

# Output the results table
print(results_df)
```

| Model               | Training Data Accuracy | Test Data Accuracy |
|---------------------|------------------------|--------------------|
| Logistic Regression | 0.8911                 | 0.8771             |
| Decision Tree       | 0.9986                 | 0.8603             |
| Random Forest       | 0.9986                 | 0.8659             |
| Naive Bayes         | 0.8715                 | 0.8324             |

Logistic Regression performed consistently across training and test datasets, indicating good generalization. Both Decision Tree and Random Forest achieved high training accuracy, but showed slightly lower test accuracy, suggesting some overfitting. Naive Bayes had the lowest overall accuracy, though still performed reasonably well for a simple model.


<a id = 'conclusion'></a>
## Conclusion

This machine learning project set out to develop predictive models to understand and anticipate employee attrition within an organization. 
By training and evaluating multiple classification algorithms—including Logistic Regression, Decision Tree, Random Forest, and Naive Bayes—we 
were able to compare their effectiveness in identifying employees at risk of leaving.

Among the models tested, Logistic Regression offered the most balanced performance, achieving high accuracy on both training and test data 
while maintaining good generalization. Decision Tree and Random Forest models achieved higher accuracy on the training data but showed signs 
of overfitting, which may limit their reliability in real-world applications. Naive Bayes, though less accurate, demonstrated consistent results 
and may still hold value in specific contexts where model simplicity and speed are essential.

Overall, the findings highlight the potential of machine learning to support human resource decision-making by identifying patterns linked to 
employee turnover. With further refinement and integration into organizational systems, these models can serve as valuable tools for early intervention 
and strategic workforce planning, ultimately contributing to improved employee retention and organizational stability.
