## 5. The Feasibility of Our Approach

In fact, our approach is aligned with the best practices available today:
- Define clear evaluation criteria (even if subjective),
- Apply them consistently,
- Use both human and AI judges,
- Try to minimize arbitrariness via detailed rubrics.

If anything, you could make it even better by:
- Validating our scoring rubric with small-scale human studies,
- Using more than one AI model as judge (ensemble judge),
- Reporting inter-rater reliability if multiple humans are scoring (e.g., using Cohen's kappa, Krippendorff’s alpha).


## 6. Data Preparation

### Data Initialization
- **X**: Feature variable from the shuffled dataframe (column "prompt").
- **y_reg**: Regression target from the shuffled dataframe (column "final score").
- **y_cls**: Classification target, obtained by rounding "final score".

### Data Splitting
- **Step 1**: Split the dataset into training set (70%) and temporary test set (30%) using `train_test_split` with `test_size=0.3` and a fixed random seed.
- **Step 2**: Further split the temporary test set into validation set and final test set, each accounting for 50% of the temporary test set, using `train_test_split` with `test_size=0.5` and the same random seed.

This process prepares the data for model training and evaluation.

### Code Components

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_excel('data.xlsx')

# Calculate final score as a weighted average of human and AI scores
df['final_score'] = 0.5 * df['human_score'] + 0.5 * df["AI_grader's_score"]

# Calculate final score as a weighted average of structure and semantic scores
df['final_score'] = 0.5 * df['structure_score'] + 0.5 * df['semantic_score']

# Shuffle the dataframe
df_shuffled = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

# Define feature and target variables
X = df_shuffled['prompt']
y_reg = df_shuffled['final_score']  # Regression task target
y_cls = df_shuffled['final_score_rounded']  # Classification task target

# Step 1: Split into train and temporary test set (70% and 30%)
X_train, X_temp, y_reg_train, y_reg_temp, y_cls_train, y_cls_temp = train_test_split(
    X, y_reg, y_cls, test_size=0.3, random_state=SEED
)

# Step 2: Split the temporary test set into validation and final test sets (50% each)
X_val, X_test, y_reg_val, y_reg_test, y_cls_val, y_cls_test = train_test_split(
    X_temp, y_reg_temp, y_cls_temp, test_size=0.5, random_state=SEED
)
```

# 10. Over-fit Prevention Summary

## Overview
Our dataset is small, so preventing over-fitting is crucial. We implement various measures across different stages of the modeling process to ensure the model generalizes well to unseen data.

## Measures by Stage

| Stage                   | Measure                         | Applies When                                                                 | Why                                                                 |
|-------------------------|---------------------------------|-----------------------------------------------------------------------------|--------------------------------------------------------------------|
| **Data Preprocessing**  | Standardization of Numeric Features | When normalizing text length features before feature extraction. | Prevents features with large scales from dominating training and aids regularization. |
| **Feature Engineering** | Feature Dimension Control       | extracting When TF-IDF features.                                            | Reduces the number of features to avoid high-dimensional overfitting, especially on small datasets. |
| **Model Training (Base Models)** | Regression Measures | When training the Logistic Regression model.                                | Keeps model complexity low, naturally reducing overfitting risk on limited data. |
|                         | Dropout Regularization          | When training the BERT-based regression model.                              | Randomly disables neurons, preventing reliance on specific activations and improving generalization. |
|                         | Weight Decay (L2 Regularization) | When optimizing the BERT regressor with AdamW.                              | Penalizes large weights, controlling model complexity and encouraging simpler models. |
|                         | Batch Normalization             | When processing hidden layers in the BERT model head.                       | Stabilizes feature distributions across mini-batches, making training smoother and less prone to overfitting. |
|                         | Gradient Clipping               | When backpropagating gradients during BERT training.                        | Prevents exploding gradients, ensuring stable updates. |
| **Model Fusion**        | OOF Prediction (Out-Of-Fold Prediction) | When generating meta-features for the stacking ensemble.                    | Ensures that each sample's prediction is made without seeing the sample during model training, strictly avoiding data leakage. |
|                         | Stacking Ensemble               | When training the meta-learner combining multiple base model outputs.       | Aggregates different models' perspectives, reducing bias and variance, thus improving robustness and preventing overfitting. |


# 11. Evaluation Method

We employ the Quadratic Weighted Kappa (QWK) metric for evaluation. This metric is highly effective for scoring tasks as it accounts for varying degrees of inconsistency and applies a more severe penalty for predictions with significant discrepancies.

（学长这里你插入一下图片）

# Conclusion


Out ensemble prompt classifier addresses a critical challenge: ensuring high-quality outputs by evaluating and optimizing user-provided prompts.

Practical Implications: It can enhance the reliability of tools like ChatGPT in real-world applications by flagging low-quality prompts.

Further Improvement: Expanding the dataset to include multilingual prompts and diverse domains. Deploying the model as a plugin for AI platforms.



   
