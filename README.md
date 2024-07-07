# 1. Problem Statement

## 1.1 Objectives
The objective of this performance study is to evaluate and compare various classification models available in sklearn using a comprehensive dataset of 2,105 patients diagnosed with Parkinson's Disease. Specifically, the study aims to achieve the following objectives:

1. Implement and benchmark multiple classification algorithms.
2. Assess the predictive accuracy, precision, recall, F1-score of each model.
3. Identify the most effective classification model for predicting Parkinson's Disease.

## 1.2 Motivation
Parkinson's Disease (PD) presents a significant public health challenge, requiring accurate diagnostic tools and predictive models to improve patient outcomes and inform clinical decision-making. This performance study is motivated by the need to systematically evaluate and compare various classification algorithms using a rich dataset encompassing diverse health-related variables. By rigorously analyzing the performance metrics of different models, we aim to:

* Provide insights into which classification algorithms are most suitable for predicting Parkinson's Disease based on different types of data.
* Enhance understanding of how preprocessing techniques impact model performance.
* Facilitate the development of robust predictive models that can assist clinicians in early diagnosis and personalized treatment strategies.
* Contribute to the broader research community by establishing best practices in applying machine learning techniques to neurological disorders.

## 1.3 Methodology
In order to analyze the performance regarding the scikit-learn models available for binary classification, we are going to use the following models:
1. Logistic Regression
2. Support Vector Machines (SVM)
3. Decision Tree
4. Random Forest
5. Gradient Boosting Machines
6. AdaBoost
7. Naïve Bayes
8. K-nearest Neighbors (KNN)
9. CatBoost Classifier
10. XGBoost Classifier

# 2. Dataset

## 2.1 Categorical and Numerical Data

- Categorical Data:
    * Discrete Categories: Categorical data represents distinct groups or categories that lack a meaningful numerical order or magnitude. Examples include gender, color, or type of car.
    * Non-Quantifiable Differences: Differences between categories cannot be quantified meaningfully. For instance, the distinction between 'red' and 'blue' lacks a numerical value.
    
    
- Numerical Data:
    * Range and Continuity: For example, the DietQuality score ranges from 0 to 10, indicating it is measured on a continuous or discrete numerical scale. Each value within this range denotes a different level of diet quality.
    * Order and Magnitude: Numerical data exhibits meaningful order and magnitude. For instance, a score of 8 signifies higher diet quality compared to a score of 5, and the difference between scores (e.g., 8 vs. 5) is quantifiable.
    
We will analyze the distribution of numerical data and the differences in their ranges.

## 2.2 Numerical Data Normalization

Since we observe numerical features with distinct values, we can apply normalization to bring these values within a closer range. We will use the `StandardScaler()` method, also known as Z-score normalization, which standardizes features by removing the mean and scaling to unit variance:

$$ z = \frac{x - \mu}{\sigma} $$

The recommended approach is to normalize the data after splitting it into training and testing sets. The rationale behind this recommendation is to prevent any information leakage from the testing set into the training set, which can lead to over-optimistic results and unrealistic performance evaluations [[1]](https://medium.com/@spinjosovsky/normalize-data-before-or-after-split-of-training-and-testing-data-7b8005f81e26).

# 3. Testing Models Performance

Metrics:

1. Accuracy:
    * Definition: Accuracy measures the proportion of correctly classified instances (both true positives and true negatives) out of the total number of instances.
    * Use Case: Useful when the classes are balanced (approximately equal number of instances for each class).
2. Precision:
    * Definition: Precision measures the proportion of true positive predictions (correctly predicted positives) out of all positive predictions made by the model.
    * Use Case: Important when the cost of false positives is high (e.g., misdiagnosing a healthy person as having Parkinson's Disease).
3. Recall (Sensitivity):
    * Definition: Recall measures the proportion of true positives (correctly predicted positives) out of all actual positive instances.
    * Use Case: Important when the cost of false negatives is high (e.g., failing to diagnose a patient who actually has Parkinson's Disease).
4. F1-score:
    * Definition: The F1-score is the harmonic mean of precision and recall. It provides a single metric that balances both precision and recall.
    * Use Case: Useful when you need to consider both false positives and false negatives equally important.

    # 4. Results

<table>
    <thead>
        <tr>
            <th>Model</th>
            <th>Accuracy</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1 Score</th>
        </tr>
    </thead>
    <tbody>
        <!--Linear Regression Results-->
        <tr>
            <td>Linear Regression</td>
            <td>0.7797</td>
            <td>0.8149</td>
            <td>0.8450</td>
            <td>0.8297</td>
        </tr>
        <!--SVM Results-->
        <tr>
            <td>Support Vector Machines</td>
            <td>0.7838</td>
            <td>0.8169</td>
            <td>0.8561</td>
            <td>0.8360</td>
        </tr>
        <!--Decision Tree Results-->
        <tr>
            <td>Decision Tree</td>
            <td>0.8741</td>
            <td>0.9129</td>
            <td>0.8893</td>
            <td>0.9009</td>
        </tr>
        <!--Random Forests Results-->
        <tr>
            <td>Random Forests</td>
            <td>0.9121</td>
            <td>0.9466</td>
            <td>0.9151</td>
            <td>0.9306</td>
        </tr>
        <!--Gradient Boosting Machines Results-->
        <tr>
            <td>Gradient Boosting Machines</td>
            <td>0.9145</td>
            <td>0.9434</td>
            <td>0.9225</td>
            <td>0.9328</td>
        </tr>
        <!--AdaBoost Results-->
        <tr>
            <td>AdaBoost</td>
            <td>0.9240</td>
            <td>0.9377</td>
            <td>0.9446</td>
            <td>0.9412</td>
        </tr>       
        <!--Naïve Bayes (GaussianNB) Results-->
        <tr>
            <td>Naïve Bayes (GaussianNB)</td>
            <td>0.7791</td>
            <td>0.8371</td>
            <td>0.8155</td>
            <td>0.8262</td>
        </tr>
        <!--K-Nearest Neighbors Results-->
        <tr>
            <td>K-Nearest Neighbors</td>
            <td>0.7316</td>
            <td>0.7862</td>
            <td>0.8007</td>
            <td>0.7934</td>
        </tr>
        <!--Category Boosting Results-->
        <tr>
            <td>Category Boosting</td>
            <td style="color:red">0.9454</td>
            <td style="color:red">0.9697</td>
            <td style="color:red">0.9446</td>
            <td style="color:red">0.9570</td>
        </tr>
        <!--eXtreme Gradient Boosting Results-->
        <tr>
            <td>eXtreme Gradient Boosting</td>
            <td>0.9145</td>
            <td>0.9572</td>
            <td>0.9077</td>
            <td>0.9318</td>
        </tr>
    </tbody>
</table>

# 5. Conclusion

## 5.1 Key Observations
* Top Performers: Models like AdaBoost, Category Boosting, Random Forests, and Gradient Boosting Machines consistently show high accuracy, precision, recall, and F1 scores. These models are robust and perform well across multiple metrics.

* Linear Regression: While it shows decent performance, its metrics are generally lower compared to the other models. Linear Regression might not capture complex relationships in the data as effectively as other models designed for classification tasks.

* K-Nearest Neighbors and Naïve Bayes: These models show lower performance metrics compared to others, which might indicate that their underlying assumptions or structure might not fit the data distribution as well in this context.

## 5.2 Final thoughts
* Model Selection: Based on these results, AdaBoost, Category Boosting, Random Forests, and Gradient Boosting Machines appear to be the top contenders for predicting Parkinson's Disease based on the given dataset.

* Further Investigation: It would be beneficial to analyze feature importance, conduct hyperparameter tuning, and potentially explore ensemble methods or feature engineering to further improve the performance of these models.