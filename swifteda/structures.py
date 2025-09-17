import pandas as pd
from IPython.display import display, HTML

class ClassificationResults:
    """A class to hold and display classification benchmark results."""
    def __init__(self, results_dict: dict):
        self.accuracy = results_dict.get('accuracy')
        self.report = results_dict.get('report')
        self.confusion_matrix_df = results_dict.get('confusion_matrix_df')

    def _repr_html_(self):
        """This magic method is called by Jupyter to display the object."""
        
        accuracy_html = f"<h4>🎯 Accuracy: {self.accuracy:.4f}</h4>"
        report_html = f"<h4>Classification Report:</h4><pre>{self.report}</pre>"
        cm_html = f"<h4>Confusion Matrix:</h4>" + self.confusion_matrix_df.to_html(classes='dataframe')
        
        tn, fp, fn, tp = self.confusion_matrix_df.values.ravel()
        class_labels = [0, 1]
        
        explanation_html = f"""
        <div style="margin-top: 10px; font-size: 0.9em; line-height: 1.6;">
        <b>What these numbers mean:</b>
        <ul>
            <li><strong>True Negatives (TN): {tn}</strong><ul><li>The model correctly predicted that <b>{tn}</b> observations belonged to <u>Class {class_labels[0]}</u>.</li></ul></li>
            <li><strong>False Positives (FP): {fp}</strong><ul><li>The model incorrectly predicted <b>{fp}</b> observations as belonging to <u>Class {class_labels[1]}</u> (when they were actually Class {class_labels[0]}).</li></ul></li>
            <li><strong>False Negatives (FN): {fn}</strong><ul><li>The model incorrectly predicted <b>{fn}</b> observations as belonging to <u>Class {class_labels[0]}</u> (when they were actually Class {class_labels[1]}).</li></ul></li>
            <li><strong>True Positives (TP): {tp}</strong><ul><li>The model correctly predicted that <b>{tp}</b> observations belonged to <u>Class {class_labels[1]}</u>.</li></ul></li>
        </ul>
        </div>
        """
        return f"{accuracy_html}{report_html}{cm_html}{explanation_html}"