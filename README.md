# Adult Income Classification

This project compares machine learning models that predict whether a person's annual income exceeded $50,000 using demographic and employment records from the UCI Adult dataset. Class `1` represents income above $50,000. Class `0` represents income of $50,000 or less.

[Open the notebook](notebook.ipynb) · [View the dataset](https://archive.ics.uci.edu/dataset/2/adult) · [Run locally](#run-locally)

## Test results

The selected model, scikit-learn's `HistGradientBoostingClassifier`, achieved a **test ROC AUC of 0.9276** on the 16,281 records in the official test split. The test results below come from the saved notebook output.

| Metric | Test score |
| --- | ---: |
| ROC AUC | 0.927583 |
| Log loss | 0.276950 |
| Precision for income above $50,000 at threshold 0.5 | 77.93% |
| Recall for income above $50,000 at threshold 0.5 | 65.37% |

At a threshold of 0.5, the model correctly identified 2,514 of the 3,846 test records in the higher-income class. Of the 3,226 records it predicted as higher-income, 2,514 belonged to that class.

## Data and preprocessing

The Adult dataset was extracted from a 1994 U.S. Census Bureau database and contributed by Barry Becker and Ronny Kohavi. The raw files and field descriptions are in [`data/`](data/). [Dataset citation and DOI](https://doi.org/10.24432/C5XW20)

| Split | Source | Records |
| --- | --- | ---: |
| Training | `adult.data` | 26,048 |
| Validation | `adult.data` | 6,513 |
| Test | `adult.test` | 16,281 |

The notebook splits `adult.data` into 80% training and 20% validation, stratified by `income` with `random_state=30`. It maps `?` to the category `unknown`, separates `income` as the target, and excludes the survey weight `fnlwgt` from the features. The reported metrics are unweighted.

During model selection, the notebook fits `DictVectorizer` on the training set and uses it to transform the validation set. Logistic regression also uses `StandardScaler` fitted on the training set's numeric features.

## Model selection

The experiments compare logistic regression, a decision tree, a random forest, histogram gradient boosting, and XGBoost against a constant baseline. The baseline predicts the training set's share of class `1` for every record. The primary selection metric is validation ROC AUC. Histogram gradient boosting scored **0.931517**, ahead of XGBoost at **0.930719** in the saved run. The [notebook](notebook.ipynb) contains the full comparison.

After model selection, the notebook refits the encoder and the selected classifier on the combined training and validation sets. It then transforms and evaluates the official `adult.test` split.

<details>
<summary>Final model configuration</summary>

```python
HistGradientBoostingClassifier(
    learning_rate=0.1,
    max_iter=100,
    max_leaf_nodes=17,
    min_samples_leaf=5,
    l2_regularization=0,
    early_stopping=False,
    random_state=30,
)
```

</details>

## Run locally

Use Python 3.12. Clone the repository and create a virtual environment.

```bash
git clone https://github.com/TgSBot-xD/adult_income_classification.git
cd adult_income_classification
python -m venv venv
```

Activate the environment with `venv\Scripts\activate.bat` in Windows Command Prompt or `source venv/bin/activate` on Linux and macOS. Install the dependencies and open the notebook from the repository root.

```bash
python -m pip install -r requirments.txt
jupyter notebook notebook.ipynb
```

Run all cells from a fresh kernel. The notebook displays the validation and test results and exports the model comparison to `data/data_models/`.
