from collections import Counter
from datasets import Dataset
from sklearn.utils import resample
import pandas as pd

def preprocess_stereoset(example):
    """
    Removes the categorization in the neutral category
    """
    sentences = []
    labels = []

    for sentence, label in zip(example['text'], example['label']):
        if label.split('_')[0]=='neutral':
          labels.append(label.split('_')[0])
        else:
          labels.append(label)
        sentences.append(sentence)

    return {'text': sentences, 'label': labels}

def label_id_mappings(data):
    """
    Create label to id, and id to label mappings for classification
    """
    ## Create label to id mappings
    label2id = {label: idx for idx, label in enumerate(sorted(set(data['label'])))}
    id2label = {idx: label for label, idx in label2id.items()}

    def convert_labels_to_ids(example):
        return {'label_id': label2id[example['label']]}

    return data.map(convert_labels_to_ids), label2id, id2label

def data_resample(dataset, resample_amount = 5000, resample_label = 'neutral', oversample = False):
   # Check class distribution
  label_counts = Counter(dataset['label'])
  print("Original label distribution:", label_counts)
  print('Type: ', type(dataset))

  # Convert the Hugging Face Dataset to a Pandas DataFrame
  df = dataset.to_pandas()

  # Separate the category to be undersampled and the rest of the data
  df_resample = df[df['label'] == resample_label]
  df_other = df[df['label'] != resample_label]

  df_resample = resample(
     df_resample,
     replace = oversample,
     n_samples = resample_amount,
     random_state = 42
  )

  df_balanced = pd.concat([df_resample, df_other])

  # Shuffle the data to mix the categories
  df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

  # Convert the balanced DataFrame back to a Hugging Face Dataset
  balanced_dataset = Dataset.from_pandas(df_balanced)
  label_counts = Counter(balanced_dataset['label'])
  print("Final label distribution:", label_counts)

  return balanced_dataset


   