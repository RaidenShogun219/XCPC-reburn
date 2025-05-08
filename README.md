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


## 


   
