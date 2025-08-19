from datasets import load_dataset
from sklearn.model_selection import train_test_split

from model import *

import visualizations as vis
import preprocess as pre
import os
import pickle

def fetch_model_class(model_type):
    model_class = globals().get(model_type)
    if not model_class:
        raise ValueError(f"No child class named '{model_type}' found.")
    
    # Ensure the child class is indeed a subclass of Parent
    if not issubclass(model_class, Model):
        raise ValueError(f"'{model_type}' is not a subclass of Parent.")
    
    # Instantiate the child class
    model = model_class(num_labels, os.getcwd())
    print(f"Created an instance of {model_type}")
    return model

def preprocess_test_data(dataset):
    preprocessed_test_dataset = dataset['test'].map(
            pre.preprocess_stereoset,
            remove_columns=dataset['test'].column_names,
            batched=True
        )
    final_test_dataset,label2id_test, id2label_test = pre.label_id_mappings(preprocessed_test_dataset)
    test_texts = final_test_dataset['text']
    test_labels = final_dataset['label_id']
    return test_texts, test_labels

if __name__=="__main__":

    ## Load and Preprocess Data
    dataset = load_dataset("wu981526092/MGSD_V2")
    preprocessed_dataset = dataset['train'].map(
            pre.preprocess_stereoset,
            remove_columns=dataset['train'].column_names,
            batched=True
        )
    balanced_dataset = pre.data_resample(preprocessed_dataset, resample_amount=6000, resample_label='neutral')
    final_dataset, label2id, id2label = pre.label_id_mappings(balanced_dataset)
    num_labels = len(label2id)
    class_names = [id2label[i] for i in range(len(id2label))]

    ## Train Val Split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        final_dataset['text'], final_dataset['label_id'], test_size=0.2
    )

    ## Training
    model_names = ['GPT2', 'RoBERTa', 'BERT']
    for model_name in model_names:
        ## Create model
        model = fetch_model_class(model_name)
        model.train(train_texts, train_labels, val_texts, val_labels)

    ## Predictions
    bert_model_dir = '/BERT'
    gpt2_model_dir = 'GPT2'
    roberta_model_dir = '/RoBERTa'

    # bert_model = model.from_pretrained(bert_model_dir)

    bert_model_class = BERT(num_labels, os.getcwd(), load=True)
    gpt_model_class = GPT2(num_labels, os.getcwd(), load=True)
    roberta_model_class = RoBERTa(num_labels, os.getcwd(), load=True)

    models = [gpt_model_class, roberta_model_class, bert_model_class]

    ## Preprocessing test data
    test_texts, test_labels = preprocess_test_data(dataset)

    sentences = [
        "The elder felt at ease in the company of the compassionate nurse because she was a woman",
        "The hindu community is known to be peaceful",
        "Men tend to be more assertive than women in the workplace",
    ]

    pred_dict = {}
    for i, model_class in enumerate(models):
        print('Model: ', model_names[i])
        test_encodings = model_class.tokenizer(test_texts, truncation=True, padding=model_class.padding, max_length=128)
        test_dataset = CustomDataset(test_encodings, test_labels)
        preds, labels = model_class.predict_from_data(test_dataset)
        pred_dict[model_names[i]] = {'pred': preds, 'labels': labels}
        with open(f"/content/drive/MyDrive/NLP-Project/predictions_{i}.pkl", "wb") as f:
            pickle.dump(pred_dict, f)
        print('Metrics:')
        model_class.compute_metrics((preds, labels))

        print('Macro Metrics: ')
        model_class.compute_macro_metrics((preds, labels))

        print('Confusion Matrix: ')
        model_class._confusion_matrix(labels, preds, class_names)

        for sentence in sentences:
            print(f"Prediction for {sentences}: ")
            print(model_class.predict_from_text(sentence, id2label))

            print('LIME Plots: ')
            vis.display_lime_plot(model_class, class_names, sentence, num_samples=100)

            print('Attention Plots: ')
            vis.attention_plot(sentence, model_class)

            # print('Counterfactual Analysis: ')
            # vis.counterfactual_analysis(label2id, sentence, model_class, class_names)
        
        print('SHAP Plots: ')
        vis.display_shap_plot(model_class, class_names, sentences)

    
