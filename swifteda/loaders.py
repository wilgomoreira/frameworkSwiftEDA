import pandas as pd
from .analyzer import DataSetAnalyzer

def load_dataset(source: str):
    """
    Loads a dataset from a local path or a URL.
    Supports .csv and .xlsx files.

    Args:
        source (str): The path or URL to the data file.
    """
    df = None  # Initialize the DataFrame as None
    try:
        if source.endswith('.csv'):
            df = pd.read_csv(source)
        elif source.endswith('.xlsx'):
            df = pd.read_excel(source)
        else:
            print(f"❌ Error: Unsupported file format. KaggleKit currently supports only '.csv' and '.xlsx'.")
            return None
        
        # The rest of the code remains the same
        if source.startswith('http'):
            display_name = source.split('/')[-1]
        else:
            display_name = source

        print(f"✅ Dataset '{display_name}' loaded successfully. It has {df.shape[0]} rows and {df.shape[1]} columns (variables).")
        
        return DataSetAnalyzer(df)
        
    except FileNotFoundError:
        print(f"Error: Local file not found at '{source}'")
        return None
    except Exception as e:
        print(f"An error occurred while trying to load from '{source}'. Details: {e}")
        return None
