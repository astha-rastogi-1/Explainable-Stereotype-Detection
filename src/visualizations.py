import torch
import shap
import lime
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from torch.nn.functional import softmax
from lime.lime_text import LimeTextExplainer
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_shap_prob(texts, model_class):
    if isinstance(texts, np.ndarray):
          texts = texts.tolist()
    tokens = model_class.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model_class.model(**tokens)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1).numpy()
    return probabilities

def predict_lime_prob(texts, model_class):
    inputs = model_class.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model_class.model(**inputs)
        logits = outputs.logits
        probabilities = softmax(logits, dim=-1).cpu().numpy()
    return probabilities

def display_shap_plot(model_class, output_labels, text):
    prediction_function = lambda texts: predict_shap_prob(texts, model_class)
    shap_explainer = shap.Explainer(prediction_function, model_class.tokenizer, output_names=output_labels)
    # Compute SHAP values
    shap_values = shap_explainer(text)
    shap.plots.text(shap_values)

def display_lime_plot(model_class, output_labels, text, num_samples=5000):
    lime_explainer = LimeTextExplainer(class_names=output_labels)
    prediction_function = lambda texts: predict_lime_prob(texts, model_class)
    exp = lime_explainer.explain_instance(text, prediction_function, num_samples=num_samples, labels=[0,1,2,3,4,5])
    exp.show_in_notebook()
    exp.save_to_file(os.path.join(os.getcwd(), f'lime_vis_{model_class.__class__.__name__}_{"-".join(text.split(" ")[:2])}.html'))

##### ATTENTION VISUALIZATION #####
def get_attention_weights(model, tokenizer, text, device):
    """
    Perform a forward pass through GPT-2 to extract attention weights.

    Args:
        model: GPT-2 model with a classification head.
        tokenizer: GPT-2 tokenizer.
        text: Input text to analyze.
        device: Device to run the model on (CPU/GPU).

    Returns:
        Tuple of (input_tokens, attentions), where:
        - input_tokens: List of tokens in the input text.
        - attentions: Attention weights from all layers and heads.
    """
    # Tokenize the input
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Forward pass with attention outputs
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    input_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze())
    attentions = outputs.attentions  # List of attention weights (layers × batch × heads × seq_len × seq_len)

    return input_tokens, attentions

def visualize_attention(attentions, input_tokens, layer=0, head=0, batch_idx=0):
    """
    Visualize attention weights for a specific layer and head.

    Args:
        attentions: Attention weights from GPT-2 (layers × batch × heads × tokens × tokens).
        input_tokens: List of input tokens.
        layer: Index of the layer to visualize.
        head: Index of the attention head to visualize.
        batch_idx: Batch index (default 0 for single input).
    """
    # Extract the attention matrix for the specified layer, head, and batch
    attention_matrix = attentions[layer][batch_idx, head, :, :].cpu().numpy()

    # Plot the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        attention_matrix,
        xticklabels=input_tokens,
        yticklabels=input_tokens,
        cmap="Blues",
        annot=False
    )
    plt.title(f"Attention Weights (Layer {layer + 1}, Head {head + 1})")
    plt.xlabel("Input Tokens")
    plt.ylabel("Input Tokens")
    plt.show()

def aggregate_and_visualize_attention(attentions, input_tokens, tokenizer, aggregation_method="mean"):
    """
    Aggregate attention across all layers and heads and visualize it.

    Args:
        attentions: List of attention weights (layers × batch × heads × tokens × tokens).
        input_tokens: List of input tokens.
        tokenizer: The tokenizer used for the model.
        aggregation_method: Aggregation method ('mean' or 'sum').

    Returns:
        Tuple of (filtered_attention_matrix, filtered_tokens).
    """
    # Convert attention tensors to a numpy array
    attentions_stacked = torch.stack(attentions)  # Shape: (layers, batch, heads, seq_len, seq_len)
    attentions_stacked = attentions_stacked.squeeze(1)  # Remove batch dimension if batch size is 1

    # Aggregate across layers and heads
    if aggregation_method == "mean":
        aggregated_attention = attentions_stacked.mean(dim=(0, 1))  # Mean over layers and heads
    elif aggregation_method == "sum":
        aggregated_attention = attentions_stacked.sum(dim=(0, 1))  # Sum over layers and heads
    else:
        raise ValueError("Invalid aggregation method. Choose 'mean' or 'sum'.")

    # Convert to numpy
    aggregated_attention = aggregated_attention.cpu().numpy()

    # Identify special tokens (e.g., START_TOKEN, END_TOKEN, PAD_TOKEN)
    special_tokens = {tokenizer.pad_token, tokenizer.eos_token, tokenizer.bos_token, "<|endoftext|>"}
    filtered_indices = [
        i for i, token in enumerate(input_tokens)
        if token not in special_tokens
    ]

    # Filter attention matrix and input tokens
    filtered_attention = aggregated_attention[np.ix_(filtered_indices, filtered_indices)]
    filtered_tokens = [input_tokens[i] for i in filtered_indices]

    # Plot heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        filtered_attention,
        xticklabels=filtered_tokens,
        yticklabels=filtered_tokens,
        cmap="Blues",
        annot=False
    )
    plt.title(f"Aggregated Attention Across All Layers and Heads ({aggregation_method.capitalize()})")
    plt.xlabel("Input Tokens")
    plt.ylabel("Input Tokens")
    plt.show()

    return filtered_attention, filtered_tokens

def attention_plot(text, model_class):
    model = model_class.model
    tokenizer = model_class.tokenizer
    model = model.to(device)

    # Get attention weights and input tokens
    input_tokens, attentions = get_attention_weights(model, tokenizer, text, device)

    # Aggregate and visualize attention without special tokens
    filtered_attention, filtered_tokens = aggregate_and_visualize_attention(
        attentions, input_tokens, tokenizer, aggregation_method="mean"
        )
    
# Function to get top 3 synonyms excluding the given word
def get_top_synonyms(word, top_n=3):
    synonyms = set()  # Use a set to avoid duplicates
    synsets = wordnet.synsets(word)
    for syn in synsets:
        for lemma in syn.lemmas():
            if lemma.name() != word:  # Exclude the word itself
                synonyms.add(lemma.name())
    return list(synonyms)[:top_n]  # Return only top N synonyms

def counterfactual_analysis(label2id, text, model_class, output_labels):
    # Assuming shap_values[0] contains the importance scores for words
    important_words = []
    probabilities = []
    prediction_function = lambda texts: predict_shap_prob(texts, model_class)
    shap_explainer = shap.Explainer(prediction_function, model_class.tokenizer, output_names=output_labels)
    # Compute SHAP values
    shap_values = shap_explainer(text)
    for i, word in enumerate(shap_values.data[0]):
        probabilities.append((word, shap_values.values[0][i][label2id['stereotype_gender']]))
    # Sort by importance and extract the most important words
    probabilities.sort(key=lambda x: x[1], reverse=True)
    important_words = [word.strip() for word, _ in probabilities[:1]]

    synonyms = get_top_synonyms(important_words[0])
    print(synonyms)

    new_Sentences = [text]
    for syn in synonyms:
        new_sent = text.replace(important_words[0], syn)
        print(new_sent)
        new_Sentences.append(new_sent)

    new_shap_values = shap_explainer(new_Sentences)
    # Visualize the explanation
    shap.plots.text(new_shap_values)
