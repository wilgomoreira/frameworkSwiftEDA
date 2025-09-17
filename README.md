# SwiftEDA: Your Data Analysis Accelerator 🚀

`SwiftEDA` is a Python framework built on `pandas` and `scikit-learn`, designed to accelerate the initial workflow of Exploratory Data Analysis (EDA), data cleaning, feature engineering, and the creation of baseline models and reports.

With a **Fluent API**, `SwiftEDA` allows you to chain operations logically and readably, turning multiple lines of complex code into a single, elegant expression.

## ✨ Key Features

  * **Fluent and Chainable API:** Write clean, readable code that flows like a story.
  * **Smart Data Loading:** Load `.csv` or `.xlsx` datasets directly from a local file or a URL.
  * **Comprehensive Analysis:** Generate full summaries, analyze numeric and categorical variables, and automatically visualize distributions and correlations.
  * **Simplified Data Cleaning:** Find and remove duplicates, and handle missing values with flexible strategies.
  * **Generalist Feature Engineering:** Create new features from numeric columns (binning) or from dates (extracting year, month, day of the week, etc.).
  * **Machine Learning Benchmark:** Run a full classification workflow with a single line of code to get a baseline model and its evaluation metrics.
  * **PDF Report Generation:** Compile the entire analysis, including plots and model results, into a professional, landscape-oriented PDF report with a single command.

## 🛠️ Installation

To use `SwiftEDA` in your projects, install it in editable mode. This makes any future improvements to the framework immediately available.

1.  Clone this repository to your local machine (or navigate to the project folder).
2.  Open a terminal in the project's root folder (`SwiftEDA_Project/`).
3.  Run the following command:
    ```bash
    pip install -e .
    ```
    This will install `SwiftEDA` and all its dependencies (`pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `openpyxl`, `Jinja2`, `WeasyPrint`).

## 🚀 Quick Start

See how easy it is to perform a full analysis, train a model, and generate a PDF report on the Titanic dataset with `SwiftEDA`. The workflow is divided into logical steps.

```python
import swifteda as sa

# --- Step 1: Load and First Look ---
# Load the data and get an initial summary to understand its raw state (e.g., find null values).
DATASET_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
analysis = sa.load_dataset(DATASET_URL)

print("--- Initial Dataset Summary (Before Cleaning) ---")
analysis.show_summary();

# --- Step 2: Data Cleaning and Feature Engineering ---
# Chain methods to clean the data and create new, valuable features.
print("\n--- Applying Cleaning and Feature Engineering Steps ---")
analysis.drop_duplicates() \
        .fill_na(strategy='median', subset=['Age']) \
        .fill_na(strategy='mode', subset=['Embarked']) \
        .bin_numeric_column('Age', 'AgeGroup', [0, 13, 20, 40, 65, 120], ['Child', 'Adolescent', 'Young Adult', 'Adult', 'Senior']);

# --- Step 3: In-depth Analysis of Cleaned Data ---
# Now, run the main analysis and plotting methods on the prepared data.
print("\n--- In-depth Analysis of the Cleaned Dataset ---")
analysis.show_summary() \
        .analyze_categoricals() \
        .plot_correlations();

# --- Step 4: Modeling and Reporting ---
# Run a Machine Learning benchmark and store the results.
print("\n--- Running Machine Learning Benchmark ---")
ml_results = analysis.run_classification_benchmark(target='Survived')

# Generate a complete PDF report, including the model results.
print("\n--- Generating Final PDF Report ---")
analysis.generate_report(
    output_path='Titanic_Analysis_Report.pdf',
    report_title="Predictive Analysis of Titanic Survival",
    ml_results=ml_results
);
```

## 📚 API Reference

Here are all the available methods you can use in `SwiftEDA`.

### Data Loading

  * `sa.load_dataset(source: str)`
      * Loads a dataset from a `.csv` or `.xlsx` file, either local or from a URL. Returns a `DataSetAnalyzer` object.

### Analysis & Visualization

  * `.show_summary()`
      * Displays a comprehensive summary of the dataset: head, tail, info, descriptive statistics, and null value counts, all formatted in tables.
  * `.plot_correlations()`
      * Plots a correlation heatmap for numeric variables and lists the top 5 strongest correlations (positive or negative).
  * `.analyze_categoricals(max_unique_values=20, figsize='auto')`
      * Analyzes all categorical columns, displaying value count tables and dynamically-sized bar charts for each.

### Data Cleaning

  * `.fill_na(strategy, subset=None)`
      * Fills null values. `strategy` can be `'mean'`, `'median'`, `'mode'`, or a specific value (e.g., 0). `subset` is an optional list of columns.
  * `.drop_na(subset=None)`
      * Removes rows containing null values.
  * `.find_duplicates()`
      * Displays all rows that are duplicates in the dataset.
  * `.drop_duplicates()`
      * Removes duplicate rows.
  * `.drop_columns(columns_to_drop: list)`
      * Removes one or more specified columns from the dataset.

### Feature Engineering

  * `.bin_numeric_column(column, new_column_name, bins, labels)`
      * Transforms a numeric column into categories (e.g., 'Age' into 'AgeGroup').
  * `.extract_datetime_features(date_column, drop_original=True)`
      * Extracts Year, Month, Day, Day of the Week, etc., from a date column.

### Modeling

  * `.run_classification_benchmark(target: str)`
      * Runs a full machine learning workflow and returns a `ClassificationResults` object with evaluation metrics.

### Reporting

  * `.generate_report(output_path, report_title, ml_results=None)`
      * Generates a complete PDF report of the entire analysis. `ml_results` is the optional object returned by `.run_classification_benchmark`.

## 🤝 How to Contribute

This is a project under development. Future contributions could include:

  * More benchmark models (Random Forest, XGBoost).
  * Functions for outlier detection and handling.
  * More data visualization options.

## 📄 License

This project is distributed under the MIT License.