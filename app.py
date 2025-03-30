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
mm23_hyperparams = hyperparameter_filters()

# Load dataset and train model
mm23_model, mm23_history, mm23_X_test, mm23_y_test = load_and_train_model("Student Depression Dataset.csv", "Depression", mm23_hyperparams)

# Display model summary
st.write("### Model Summary")
mm23_summary_string = []
mm23_model.summary(print_fn=lambda x: mm23_summary_string.append(x), line_length=1000)
st.code("\n".join(mm23_summary_string), language="plaintext")

# Display training history
st.write("### Training History")

# Convert history to DataFrame
mm23_history_df = pd.DataFrame(mm23_history.history)
st.line_chart(mm23_history_df)

# Load test dataset for evaluation
st.write("### Model Evaluation")

# Get model predictions
mm23_y_pred = mm23_model.predict(mm23_X_test)

# Convert predictions to binary format
mm23_y_pred_binary = (mm23_y_pred > 0.5).astype(int)

# Compute confusion matrix
mm23_cm = confusion_matrix(mm23_y_test, mm23_y_pred_binary)

# Plot Confusion Matrix
st.write("### Confusion Matrix")
mm23_fig, mm23_ax = plt.subplots()
sns.heatmap(mm23_cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Student Depression", "Student Depression"], yticklabels=["No Student Depression", "Student Depression"])
mm23_ax.set_xlabel("Predicted Label")
mm23_ax.set_ylabel("True Label")
st.pyplot(mm23_fig)

# Compute classification report
st.write("### Classification Report")
mm23_report = classification_report(mm23_y_test, mm23_y_pred_binary, output_dict=True)
st.dataframe(pd.DataFrame(mm23_report).transpose())

# Compute ROC Curve
mm23_fpr, mm23_tpr, _ = roc_curve(mm23_y_test, mm23_y_pred)
mm23_roc_auc = auc(mm23_fpr, mm23_tpr)

# Plot ROC Curve
st.write("### ROC Curve")
mm23_fig_roc, mm23_ax_roc = plt.subplots()
mm23_ax_roc.plot(mm23_fpr, mm23_tpr, color='blue', lw=2, label=f'AUC = {mm23_roc_auc:.2f}')
mm23_ax_roc.plot([0, 1], [0, 1], color='gray', linestyle='--')
mm23_ax_roc.set_xlabel("False Positive Rate")
mm23_ax_roc.set_ylabel("True Positive Rate")
mm23_ax_roc.set_title("Receiver Operating Characteristic (ROC) Curve")
mm23_ax_roc.legend(loc="lower right")
st.pyplot(mm23_fig_roc)