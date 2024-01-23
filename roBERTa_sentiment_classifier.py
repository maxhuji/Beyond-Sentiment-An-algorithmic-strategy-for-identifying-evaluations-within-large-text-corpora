# -*- coding: utf-8 -*-
"""roBERTa_sentiment.ipynb

This script was used via a google colab notebook - https://colab.research.google.com/.


"""

!pip install simpletransformers
import pandas as pd
from simpletransformers.classification import ClassificationModel

# load pre-trained
# use_cuda = True to load to GPU
model = ClassificationModel('roberta', 'cardiffnlp/twitter-roberta-base-sentiment-latest', num_labels=3, use_cuda=True)

df = pd.read_excel("Classifications_Output.xlsx")
df["text"] = df["SNIPPET"]

df.shape

# remove [+] and [-] from text column 
df["text"] = df["text"].str.replace("[+]", "", regex=False)
df["text"] = df["text"].str.replace("[-]", "", regex=False)

# remove [[ and  ]] from text column
df["text"] = df["text"].str.replace("[[", "", regex=False)
df["text"] = df["text"].str.replace("]]", "", regex=False)

#predict sentiment values based on the pre-trained model
preds = model.predict(df["text"].to_list())

#add predictions to new column in dataframe
df['roBERTa_sentiment'] = preds[0]

#export
df.to_excel("Classifications_Output_v2.xlsx")