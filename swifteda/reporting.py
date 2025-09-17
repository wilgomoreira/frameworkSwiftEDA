import os
import base64
from io import BytesIO, StringIO
import datetime
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML as WeasyHTML
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from .structures import ClassificationResults

def create_pdf_report(dataframe, output_path: str, report_title="Data Analysis", ml_results=None):
    """
    Orchestrates data collection and generates a complete PDF report.

    Args:
        dataframe (pd.DataFrame): The DataFrame with the data to be analyzed.
        output_path (str): The path to save the PDF file.
        report_title (str): The title to be displayed in the report.
        ml_results (ClassificationResults, optional): Object with ML benchmark results.
    """
    print(f"📄 Generating PDF report at '{output_path}'...")
    
    # --- 1. Prepare Templating Environment ---
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template('report_template.html')

    # --- 2. Collect All Data for the Report ---
    context = {
        'report_title': report_title,
        'generation_date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'categorical_analysis': [],
        'correlation_plot_b64': None,
        'correlation_table': None,
        'classification_results': None
    }
    
    # --- 2a. Summary Data ---
    info_buffer = StringIO()
    dataframe.info(buf=info_buffer)
    context['summary_info'] = info_buffer.getvalue()
    context['summary_describe_html'] = dataframe.describe().to_html(classes='dataframe')
    null_counts = dataframe.isnull().sum().to_frame('null_count')
    context['summary_nulls_html'] = null_counts[null_counts['null_count'] > 0].to_html(classes='dataframe')

    # --- 2b. Categorical Analysis ---
    categorical_cols = dataframe.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if dataframe[col].nunique() <= 20:
            vc = dataframe[col].value_counts()
            
            num_categories = len(vc)
            width = max(8, num_categories * 0.6)
            height = 5
            
            fig, ax = plt.subplots(figsize=(width, height))
            sns.barplot(x=vc.index, y=vc.values, hue=vc.index, palette='viridis', legend=False, ax=ax)
            ax.set_title(f"Distribution of '{col}'")
            ax.tick_params(axis='x', rotation=45)
            fig.tight_layout()
            
            img_buffer = BytesIO()
            fig.savefig(img_buffer, format='png', bbox_inches='tight')
            plt.close(fig)
            
            context['categorical_analysis'].append({
                'name': col,
                'table': vc.to_frame().to_html(classes='dataframe'),
                'plot_b64': base64.b64encode(img_buffer.getvalue()).decode()
            })

    # --- 2c. Correlation Analysis ---
    numeric_df = dataframe.select_dtypes(include=['number'])
    if numeric_df.shape[1] >= 2:
        corr_matrix = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        
        img_buffer = BytesIO()
        fig.savefig(img_buffer, format='png', bbox_inches='tight')
        plt.close(fig)
        context['correlation_plot_b64'] = base64.b64encode(img_buffer.getvalue()).decode()

        corr_pairs = corr_matrix.unstack().sort_values(key=abs, ascending=False)
        corr_pairs = corr_pairs[corr_pairs < 1.0]
        top_5_pairs = corr_pairs.iloc[::2].head(5)
        top_corr_df = top_5_pairs.reset_index()
        top_corr_df.columns = ['Variable 1', 'Variable 2', 'Correlation']
        top_corr_df['Correlation'] = top_corr_df['Correlation'].map('{:+.4f}'.format)
        context['correlation_table'] = top_corr_df.to_html(classes='dataframe', index=False)
    
    # --- 2d. Machine Learning Results ---
    if ml_results and isinstance(ml_results, ClassificationResults):
        context['classification_results'] = {
            'report': ml_results.report,
            'confusion_matrix_html': ml_results.confusion_matrix_df.to_html(classes='dataframe')
        }
    
    # --- 3. Render HTML with Data ---
    html_out = template.render(context)

    # --- 4. Convert HTML to PDF ---
    WeasyHTML(string=html_out, base_url=os.path.dirname(__file__)).write_pdf(output_path)
    
    print(f"✅ Report successfully saved at '{output_path}'!")