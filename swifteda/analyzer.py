from . import plotting
from IPython.display import display, HTML
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import base64
from io import BytesIO
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML as WeasyHTML
import datetime
from . import reporting
from .structures import ClassificationResults

class DataSetAnalyzer:
    def __init__(self, dataframe):
        self._df = dataframe

    def show_summary(self):
        """Shows a complete and well-formatted summary of the dataset."""
        display(HTML('<h2>Dataset Summary</h2>'))

        display(HTML('<h3>First 5 Rows</h3>'))
        display(self._df.head())

        display(HTML('<h3>Last 5 Rows</h3>'))
        display(self._df.tail())

        display(HTML('<h3>General Info and Data Types</h3>'))
        self._df.info()

        display(HTML('<h3>Null Value Count</h3>'))
        null_counts = self._df.isnull().sum().to_frame('null_count')
        display(null_counts[null_counts['null_count'] > 0])

        display(HTML('<h3>Descriptive Statistics (Numeric Variables)</h3>'))
        display(self._df.describe())

        return self

    def find_duplicates(self):
        """
        Finds and displays any duplicate rows in the dataset.
        """
        duplicate_rows = self._df[self._df.duplicated(keep=False)]

        display(HTML('<h3>Duplicate Rows Check</h3>'))

        if not duplicate_rows.empty:
            print(f"🚨 Found {len(duplicate_rows)} rows that are part of duplicate sets.")
            display(duplicate_rows.sort_values(by=list(duplicate_rows.columns)))
        else:
            print("✅ No duplicate rows found in the dataset.")

        return self

    def drop_duplicates(self):
        """
        Removes duplicate rows from the dataset, keeping the first occurrence.
        """
        initial_rows = len(self._df)
        self._df.drop_duplicates(inplace=True)
        rows_dropped = initial_rows - len(self._df)

        if rows_dropped > 0:
            print(f"\n🗑️ {rows_dropped} duplicate row(s) removed. {len(self._df)} rows remain.")
        else:
            print("\nℹ️ No action needed, no duplicate rows to remove.")

        return self

    def drop_na(self, subset=None):
        """
        Removes rows with null (NaN) values.

        Args:
            subset (list, optional): List of columns to consider for NaNs.
                                     If None, considers all columns. Defaults to None.
        """
        initial_rows = len(self._df)
        self._df.dropna(subset=subset, inplace=True)
        rows_dropped = initial_rows - len(self._df)

        if rows_dropped > 0:
            subset_str = f" in columns {subset}" if subset else ""
            print(f"\n🗑️  {rows_dropped} row(s) with null values removed{subset_str}.")

        return self

    def drop_columns(self, columns_to_drop: list):
        self._df.drop(columns=columns_to_drop, inplace=True)
        print(f"\n🗑️ Column(s) {columns_to_drop} removed.")
        return self

    def fill_na(self, strategy, subset=None):
        """
        Fills null (NaN) values using a specific strategy.

        Args:
            strategy (str or value): The strategy to use. Can be 'mean', 'median', 'mode'
                                     or a specific value (e.g., 0, "Unknown").
            subset (list, optional): List of columns to apply the strategy.
                                     If None, applies to all possible columns. Defaults to None.
        """
        columns_to_fill = subset or self._df.columns

        for column in columns_to_fill:
            if self._df[column].isnull().any():
                fill_value = None

                if strategy == 'mean':
                    if pd.api.types.is_numeric_dtype(self._df[column]):
                        fill_value = self._df[column].mean()
                        self._df[column] = self._df[column].fillna(fill_value)
                        print(f"\n✏️  Null values in column '{column}' filled with mean ({fill_value:.2f}).")

                elif strategy == 'median':
                    if pd.api.types.is_numeric_dtype(self._df[column]):
                        fill_value = self._df[column].median()
                        self._df[column] = self._df[column].fillna(fill_value)
                        print(f"\n✏️  Null values in column '{column}' filled with median ({fill_value:.2f}).")

                elif strategy == 'mode':
                    fill_value = self._df[column].mode()[0]
                    self._df[column] = self._df[column].fillna(fill_value)
                    print(f"\n✏️  Null values in column '{column}' filled with mode ('{fill_value}').")

                else:
                    fill_value = strategy
                    self._df[column] = self._df[column].fillna(fill_value)
                    print(f"\n✏️  Null values in column '{column}' filled with value '{fill_value}'.")

        return self

    def analyze_categoricals(self, max_unique_values=20, figsize='auto'):
        """
        Analyzes and visualizes the categorical columns of the dataset.

        Args:
            max_unique_values (int): Limit of unique values to plot.
            figsize (tuple or str): Size of bar charts. 'auto' for dynamic.
        """
        display(HTML('<h2>Categorical Variable Analysis</h2>'))

        categorical_cols = self._df.select_dtypes(include=['object', 'category']).columns

        if len(categorical_cols) == 0:
            display(HTML('<p>No categorical variables found in the dataset.</p>'))
            return self

        for column in categorical_cols:
            num_unique = self._df[column].nunique()

            display(HTML(f"<h4>Column: '{column}'</h4>"))
            print(f"   (Found {num_unique} unique values)")

            if num_unique <= max_unique_values:
                value_counts = self._df[column].value_counts()
                display(value_counts.to_frame())

                plotting.plot_bar_chart(value_counts, f"Distribution of Column '{column}'", figsize=figsize)
            else:
                print(f"   -> Too many unique values to display and plot. Skipping chart.")

            display(HTML("<hr>"))

        return self

    def bin_numeric_column(self, column: str, new_column_name: str, bins: list, labels: list):
        """
        Creates a new categorical column from a numeric column,
        grouping values into bins.

        Args:
            column (str): Name of the numeric column to transform.
            new_column_name (str): Name of the new categorical column to create.
            bins (list): List of numbers defining bin edges.
                         Ex: [0, 18, 65, 100]
            labels (list): List of strings with labels for each bin.
                           Must have one less item than bins.
                           Ex: ['Young', 'Adult', 'Senior']
        """
        if column not in self._df.columns:
            print(f"⚠️ Error: Column '{column}' not found in the dataset.")
            return self
        if not pd.api.types.is_numeric_dtype(self._df[column]):
            print(f"⚠️ Error: Column '{column}' is not numeric.")
            return self
        if len(bins) != len(labels) + 1:
            print(f"⚠️ Error: Number of labels must be one less than number of bin edges.")
            return self

        self._df[new_column_name] = pd.cut(
            x=self._df[column],
            bins=bins,
            labels=labels,
            right=False,
            include_lowest=True
        )

        print(f"🛠️  Feature '{new_column_name}' created from column '{column}'.")

        return self

    def extract_datetime_features(self, date_column: str, drop_original=True):
        """
        Extracts multiple features from a date/time column.

        Converts the specified column to datetime and creates new columns
        for Year, Month, Day, Day of Week, and Week of Year.

        Args:
            date_column (str): Name of the column containing dates.
            drop_original (bool): If True, the original date column will be removed.
        """
        if date_column not in self._df.columns:
            print(f"⚠️ Error: Column '{date_column}' not found in the dataset.")
            return self

        try:
            self._df[date_column] = pd.to_datetime(self._df[date_column])
        except Exception as e:
            print(f"⚠️ Error converting column '{date_column}' to datetime. Details: {e}")
            return self

        self._df[f'{date_column}_Year'] = self._df[date_column].dt.year
        self._df[f'{date_column}_Month'] = self._df[date_column].dt.month
        self._df[f'{date_column}_Day'] = self._df[date_column].dt.day
        self._df[f'{date_column}_DayOfWeek'] = self._df[date_column].dt.dayofweek  # Monday=0, Sunday=6
        self._df[f'{date_column}_WeekOfYear'] = self._df[date_column].dt.isocalendar().week

        print(f"🛠️  Datetime features extracted from column '{date_column}'.")

        # Remove the original column if requested
        if drop_original:
            self.drop_columns([date_column])

        return self

    def plot_distributions(self, save_path=None):
        """Plots the distribution of all numeric columns."""
        plotting.plot_histograms(self._df, save_path)
        return self

    def plot_correlations(self, save_path=None):
        """
        Plots the correlation matrix and displays a formatted table of the top 5 correlations.
        """
        print("\nGenerating correlation heatmap...")

        numeric_df = self._df.select_dtypes(include=['number'])
        if numeric_df.shape[1] < 2:
            print("Not enough numeric columns for a correlation map.")
            return self

        corr_matrix = numeric_df.corr()
        plotting.plot_correlation_heatmap(corr_matrix, save_path)

        corr_pairs = corr_matrix.unstack().sort_values(key=abs, ascending=False)
        corr_pairs = corr_pairs[corr_pairs < 1.0]
        top_5_pairs = corr_pairs.iloc[::2].head(5)

        if not top_5_pairs.empty:
            top_corr_df = top_5_pairs.reset_index()
            top_corr_df.columns = ['Variable 1', 'Variable 2', 'Correlation']
            top_corr_df['Correlation'] = top_corr_df['Correlation'].map('{:+.4f}'.format)
            display(HTML('<h3>Top 5 Highest Absolute Correlations</h3>'))
            display(top_corr_df)
        else:
            display(HTML('<p>No significant correlations found.</p>'))

        return self

    def run_classification_benchmark(self, target: str, model='LogisticRegression'):
        """
        Runs a complete and basic ML classification workflow.

        Args:
            target (str): The name of the target column (y) for prediction.
            model (str, optional): Currently supports 'LogisticRegression'. Defaults to 'LogisticRegression'.
        """
        display(HTML(f"<h2>🚀 Running Classification Benchmark for Target: '{target}'</h2>"))

        # 1. Separate Features (X) and Target (y)
        if target not in self._df.columns:
            print(f"❌ Error: Target column '{target}' not found.")
            return self

        df_cleaned = self._df.dropna(subset=[target])
        X = df_cleaned.drop(target, axis=1)
        y = df_cleaned[target]

        # 2. Define Automatic Preprocessing
        numeric_features = X.select_dtypes(include=['number']).columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])

        # 3. Split into Train and Test Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        print("📊 Data split: 80% for training, 20% for testing.")

        # 4. Create and Train the Pipeline
        if model == 'LogisticRegression':
            clf = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', LogisticRegression(random_state=42, max_iter=1000))])
        else:
            print(f"❌ Model '{model}' not supported. Using LogisticRegression by default.")
            clf = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', LogisticRegression(random_state=42, max_iter=1000))])

        print("⚙️  Training the model...")
        clf.fit(X_train, y_train)
        print("✅ Model trained successfully.")

        # 5. GENERATE AND RETURN RESULTS (NO LONGER DISPLAY)
        y_pred = clf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        class_labels = sorted(y.unique())
        cm_df = pd.DataFrame(cm,
                             index=[f'Real Class {class_labels[0]}', f'Real Class {class_labels[1]}'],
                             columns=[f'Predicted Class {class_labels[0]}', f'Predicted Class {class_labels[1]}'])

        # Create the results dictionary
        results = {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix_df': cm_df  # Pass the formatted DataFrame
        }

        print("🏁 Benchmark completed. Results returned.")
        return ClassificationResults(results)

    def generate_report(self, output_path: str, report_title="Data Analysis", ml_results=None):
        """
        Generates a complete PDF report.

        Args:
            output_path (str): Path to save the PDF.
            report_title (str): Title of the report.
            ml_results (dict, optional): Dictionary with ML benchmark results.
        """
        reporting.create_pdf_report(
            dataframe=self._df,
            output_path=output_path,
            report_title=report_title,
            ml_results=ml_results
        )
        return self
    