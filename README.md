# Explainable Stereotype Detection in Text

This project builds an AI system to detect stereotypes in text while ensuring explainability so decisions are transparent and trustworthy. It combines state-of-the-art transformer models with interpretable AI methods to make model predictions understandable to both researchers and non-technical users.

## 🔑 Highlights

- Fine-tuned BERT, RoBERTa, and GPT-2 for stereotype detection.
- Applied explainability methods (LIME, SHAP, attention heatmaps) to show why a model made a prediction.
- Built a robust evaluation pipeline (accuracy, F1, precision, recall, confusion matrix).
- Designed for AI fairness and bias research, with applications in NLP ethics.

## 💡 Example Use Case

Input:
```
"Men tend to be more assertive than women in the workplace."
```


Output:

- Predicted label: Gender Stereotype

- Highlighted words influencing the decision (via LIME & SHAP).

## 🌍 Impact

This project helps make bias detection in text more transparent by combining classification accuracy with clear explanations. It can be extended to support fairness auditing, educational tools, and multilingual bias research.

## Features

- Fine-tunes multiple transformer models (BERT, RoBERTa, GPT-2, optional ALBERT and T5) for stereotype detection.

- Preprocessing pipeline for balancing and mapping labels in the MGSD_V2 dataset.

- Model evaluation with accuracy, F1-score, precision, recall, and confusion matrices.

Explainability module:

- LIME plots to show local feature importance.

- SHAP plots for global interpretability across examples.

- Attention heatmaps for word-level attention visualization.


## 📊 Dataset
This project uses the MGSD_V2 dataset (Multicultural Gender Stereotype Dataset).
It contains labeled text for stereotype classification.

## 🚀 Usage
**Training**

Run the main script to train and evaluate models:

```bash
python main.py
```
This will:

- Load and preprocess the dataset.

- Train BERT, RoBERTa, GPT-2 on stereotype classification.

- Save trained models to respective directories.

- Evaluate models on validation/test sets.

**Inference & Explainability**

Modify the sentences list in main.py with your own examples.
The script will generate:

- Predictions per model.

- LIME and SHAP plots.

- Attention heatmaps.

Example output for a sentence:

```vbnet
Sentence: "Men tend to be more assertive than women in the workplace."
Prediction: stereotype → gender
```
## 📈 Evaluation Metrics
- Accuracy, Precision, Recall, F1-score (weighted & macro)

- Confusion Matrix visualization for class-level insights

## 🔍 Explainability Examples
LIME → highlights words influencing stereotype detection

SHAP → shows global feature importance across the dataset

Attention Maps → visualize which tokens the transformer attends to

Counterfactual Analysis → test model robustness (optional)

