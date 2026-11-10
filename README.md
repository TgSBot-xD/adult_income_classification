# Adult Income Classification

This project compares binary classifiers on the UCI Adult dataset. The task is to predict whether a person's annual income exceeded $50,000 from the attributes in their record. Class `1` represents income above $50,000, and class `0` represents income of $50,000 or less.

## Dataset and splits

Barry Becker and Ronny Kohavi contributed the Adult dataset, which was extracted from a 1994 U.S. Census Bureau database. The data files and field descriptions are in [`data/`](data/). [UCI Adult dataset](https://archive.ics.uci.edu/dataset/2/adult) · [DOI 10.24432/C5XW20](https://doi.org/10.24432/C5XW20)

| Split      | Source       | Records |
| ---------- | ------------ | ------: |
| Training   | `adult.data` |  26,048 |
| Validation | `adult.data` |   6,513 |
| Test       | `adult.test` |  16,281 |

The notebook splits `adult.data` into 80% training and 20% validation, stratified by `income` with `random_state=30`. It uses the validation set to compare models and tune hyperparameters. The selected model is evaluated on `adult.test`.

The notebook maps missing values marked `?` to the category `unknown` and separates `income` as the target. It excludes the survey weight `fnlwgt` from the features and computes unweighted metrics. It fits `DictVectorizer` on the training set and uses it to transform the validation set. For logistic regression, it also scales numeric features with `StandardScaler` fitted on the training set.

## Model comparison

**Validation ROC AUC** is the primary model selection metric. The constant baseline assigns every record the same probability, equal to the share of class `1` in the training set. These results come from the saved output in the [notebook](notebook.ipynb).

| Model                              | Validation ROC AUC ↑ | Validation log loss ↓ |
| ---------------------------------- | -------------------: | --------------------: |
| Constant baseline                  |             0.500000 |              0.551942 |
| Logistic Regression                |             0.907617 |              0.319017 |
| Decision Tree                      |             0.909857 |              0.315484 |
| Random Forest                      |             0.921525 |              0.295463 |
| XGBoost                            |             0.930719 |              0.274643 |
| **HistGradientBoostingClassifier** |         **0.931517** |          **0.274435** |

## Final evaluation

`HistGradientBoostingClassifier` achieved the highest validation ROC AUC. The notebook refits the encoder and the selected model on the combined training and validation sets, then uses that encoder to transform the test set.

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

| Test metric                                 |    Score |
| ------------------------------------------- | -------: |
| ROC AUC                                     | 0.927583 |
| Log loss                                    | 0.276950 |
| Class `1` precision at a threshold of `0.5` |   77.93% |
| Class `1` recall at a threshold of `0.5`    |   65.37% |

The test confusion matrix uses a threshold of `0.5`. Rows are actual classes and columns are predicted classes.

| Actual class | Predicted `0` | Predicted `1` |
| ------------ | ------------: | ------------: |
| `0`          |        11,723 |           712 |
| `1`          |         1,332 |         2,514 |

## Run the notebook

Use Python 3.12. Clone the repository and create a virtual environment from its root directory.

```bash
git clone https://github.com/TgSBot-xD/adult_income_classification.git
cd adult_income_classification
python -m venv venv
```

Activate the environment with `venv\Scripts\activate.bat` in Windows Command Prompt or `source venv/bin/activate` on Linux and macOS. Then install the dependencies and open the notebook.

```bash
python -m pip install -r requirments.txt
jupyter notebook notebook.ipynb
```

Start with a fresh kernel and run all cells in order. The notebook displays the validation and test results and exports the model comparison as a CSV file in `data/data_models/`.
