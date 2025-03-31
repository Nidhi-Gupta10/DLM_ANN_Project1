import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from Filter import hyperparameter_filters
from Connector import load_and_train_model
from ModelSummary import model_summary_to_df

st.set_page_config(page_title="Student Depression ANN Dashboard", layout="wide")

st.title("Student Depression with ANN 🏥")

# Get hyperparameter selections from sidebar
ng_06_hyperparams = hyperparameter_filters()

# Load dataset and train model
ng_06_model, ng_06_history, ng_06_X_test, ng_06_y_test = load_and_train_model("Student Depression Dataset.csv", "Depression", ng_06_hyperparams)

# Display model summary
st.write("### Model Summary")
ng_06_summary_string = []
ng_06_model.summary(print_fn=lambda x: ng_06_summary_string.append(x), line_length=1000)
st.code("\n".join(ng_06_summary_string), language="plaintext")

# Display training history
st.write("### Training History")

# Convert history to DataFrame
ng_06_history_df = pd.DataFrame(ng_06_history.history)
st.line_chart(ng_06_history_df)

# Load test dataset for evaluation
st.write("### Model Evaluation")

# Get model predictions
ng_06_y_pred = ng_06_model.predict(ng_06_X_test)

# Convert predictions to binary format
ng_06_y_pred_binary = (ng_06_y_pred > 0.5).astype(int)

# Compute confusion matrix
ng_06_cm = confusion_matrix(ng_06_y_test, ng_06_y_pred_binary)

# Plot Confusion Matrix
st.write("### Confusion Matrix")
ng_06_fig, ng_06_ax = plt.subplots()
sns.heatmap(ng_06_cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Student Depression", "Student Depression"], yticklabels=["No Student Depression", "Student Depression"])
ng_06_ax.set_xlabel("Predicted Label")
ng_06_ax.set_ylabel("True Label")
st.pyplot(ng_06_fig)

# Compute classification report
st.write("### Classification Report")
ng_06_report = classification_report(ng_06_y_test, ng_06_y_pred_binary, output_dict=True)
st.dataframe(pd.DataFrame(ng_06_report).transpose())

# Compute ROC Curve
ng_06_fpr, ng_06_tpr, _ = roc_curve(ng_06_y_test, ng_06_y_pred)
ng_06_roc_auc = auc(ng_06_fpr, ng_06_tpr)

# Plot ROC Curve
st.write("### ROC Curve")
ng_06_fig_roc, ng_06_ax_roc = plt.subplots()
ng_06_ax_roc.plot(ng_06_fpr, ng_06_tpr, color='blue', lw=2, label=f'AUC = {ng_06_roc_auc:.2f}')
ng_06_ax_roc.plot([0, 1], [0, 1], color='gray', linestyle='--')
ng_06_ax_roc.set_xlabel("False Positive Rate")
ng_06_ax_roc.set_ylabel("True Positive Rate")
ng_06_ax_roc.set_title("Receiver Operating Characteristic (ROC) Curve")
ng_06_ax_roc.legend(loc="lower right")
st.pyplot(ng_06_fig_roc)
