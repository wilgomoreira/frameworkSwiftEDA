import matplotlib.pyplot as plt
import seaborn as sns

def plot_histograms(df, save_path=None):
    """Creates and shows/saves histograms for numeric columns."""
    numeric_df = df.select_dtypes(include=['number'])
    if numeric_df.empty:
        print("No numeric columns to plot.")
        return

    numeric_df.hist(bins=15, figsize=(15, 10), layout=(-1, 4))
    plt.suptitle("Frequency Distribution of Numeric Variables")

    if save_path:
        plt.savefig(save_path)
        print(f"Distribution plots saved to '{save_path}'")
    else:
        plt.show()

def plot_correlation_heatmap(corr_matrix, save_path=None):
    """Creates and shows/saves a correlation heatmap."""
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title("Correlation Matrix")
    
    if save_path:
        plt.savefig(save_path)
        print(f"Correlation heatmap saved to '{save_path}'")
    else:
        plt.show()

def plot_bar_chart(data, title, figsize='auto'):
    """
    Creates and displays a bar chart with dynamic or fixed size.

    Args:
        data (pd.Series): Data to plot (index=categories, values=count).
        title (str): Chart title.
        figsize (tuple or str): If 'auto', size is calculated dynamically.
                                If a tuple (width, height), uses fixed size.
    """
    if figsize == 'auto':
        # Dynamic sizing logic:
        # Width increases with number of categories to avoid overlap.
        num_categories = len(data)
        width = max(4, num_categories * 0.1) # 0.6" per bar, minimum 8"
        height = 4
        calculated_figsize = (width, height)
    else:
        calculated_figsize = figsize

    plt.figure(figsize=calculated_figsize)
    sns.barplot(x=data.index, y=data.values, hue=data.index, palette='viridis', legend=False)
    plt.title(title, fontsize=16)
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()