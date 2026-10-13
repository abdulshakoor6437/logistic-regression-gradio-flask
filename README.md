# Logistic Regression with Gradio and Flask UI

Logistic regression classifiers on two Kaggle datasets (Water Potability and Heart Failure), with interactive web interfaces built in **Gradio** and **Flask**. Developed and run on Google Colab.

## Datasets

| # | Dataset | Target | Task |
|---|---------|--------|------|
| 1 | [Water Potability](https://www.kaggle.com/datasets/adityakadiwal/water-potability) | `Potability` | Is the water safe to drink? (0 = Not Potable, 1 = Potable) |
| 2 | [Heart Failure Clinical Records](https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data) | `DEATH_EVENT` | Will the patient die during follow-up? (0 = Survived, 1 = Death Event) |

## Project Workflow

1. Download both datasets from Kaggle and load them with pandas
2. Explore the data (shape, class balance, missing values)
3. Build a scikit-learn pipeline for each dataset:
   - `SimpleImputer` (median) to fill missing values
   - `StandardScaler` to scale features
   - `LogisticRegression` with `class_weight='balanced'`
4. Split the data 80/20 (stratified) and evaluate with accuracy, confusion matrix and classification report
5. Save each trained model with `joblib`
6. Serve predictions through two web UIs

## User Interfaces

**Gradio**
- One tab per dataset
- Sliders for numeric inputs, radio buttons for yes/no inputs
- Colored result card with probability bars and a reset button

**Flask**
- Dataset tabs with a responsive input form
- Typical-range hints for every field
- Colored result card with confidence and probability bars

## Tech Stack

Python, pandas, scikit-learn, joblib, Gradio, Flask, Google Colab

## How to Run

1. Open `logistic_regression_gradio_flask.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Download the two CSV files from the Kaggle links above and upload them to the Colab session (or use the Kaggle API)
3. Run the cells from top to bottom
4. Open the Gradio link or the embedded Flask page to try the predictions

## Notes

- Missing values (in `ph`, `Sulfate` and `Trihalomethanes` of the water dataset) are filled with the column median inside the pipeline, fitted on the training data only to avoid data leakage.
- The Heart Failure dataset has no missing values.
- Do not upload your `kaggle.json` API key to this repository.

## Author

Abdul Shakoor - Szabist University / ANN
