# -*- coding: utf-8 -*-
"""roBERTa_evaluations.ipynb

This script was used via a google colab notebook - https://colab.research.google.com/

"""

!pip install simpletransformers

import pandas as pd
import numpy as np
from simpletransformers.classification import ClassificationModel
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

# load file after initial preprocessing steps in R (done to Q1 and Q2 only )
df = pd.read_excel('Classifications_Output.xlsx')
df

# remove rows where IS_GOLD_STANDARD = YES
df = df[df['IS_GOLD_STANDARD'] == "NO"]
df.shape

df["Q2"] = df["Q2"]. replace(np. nan,0)
df["Q2"] = df["Q2"].replace(3,0)
df["Q2"] = df["Q2"].replace(4,0)

# rename columns
df["text"] = df["SNIPPET"]
df["labels"] = df["Q2"]

# Function to replace the token before the sentiment word with the Q1 info
def replace_token_with_number(row):
    string = row['SNIPPET']
    number = row['Q1']
    return string.replace('[[', "xxproj " + str(number) + " ")

# Apply the function to every row in the dataframe
df["text"] = df.apply(replace_token_with_number, axis=1)

# remove [+] and [-] from text column and replace with ++ and --
df["text"] = df["text"].str.replace("[+]", " xxpositive", regex=False)
df["text"] = df["text"].str.replace("[-]", " xxnegative", regex=False)

# remove [[ and  ]] from text column
df["text"] = df["text"].str.replace("[[", "", regex=False)
df["text"] = df["text"].str.replace("]]", "", regex=False)

set(df["labels"])

df['Q2'].value_counts()

# take a look at the first text after the above changes
df["text"].iloc[1]

# calculate class weights for labels column
class_weights = compute_class_weight('balanced', classes=np.unique(df["labels"]), y=df['labels'])
class_weights

# create reduced dataset for k-fold cross validation
df_new = df[["text","labels"]]
df_new

#run 10-fold cross validation
from simpletransformers.classification import ClassificationModel
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd

# prepare cross validation
n=10
kf = KFold(n_splits=n, shuffle=True)

results = []

# initiate roberta-base model for predicting 3 multiclass categories, with the class weights calculated above.
# use_cuda=True for running on GPU
# training for 10 epochs - try other values too!

for train_index, test_index in kf.split(df_new):
  # splitting Dataframe (dataset  not included)
    train_df = df_new.iloc[train_index]
    test_df = df_new.iloc[test_index]
    # Defining Model
    model = ClassificationModel('roberta', 'roberta-base', num_labels=3, weight=class_weights.tolist(),
                            use_cuda=True, args={'reprocess_input_data': True, 'overwrite_output_dir': True,
                                                  "num_train_epochs": 10})
    
  # train the model
    model.train_model(train_df)
  # validate the model 
    result, model_outputs, wrong_predictions = model.eval_model(test_df, f1=classification_report, conf=confusion_matrix)
    print(result['f1'])
    print(result['conf'])
  # append model score
    results.append(result['f1'])

#train model on full dataset of individual coding for subsequent comparison with gold standard data

#initiate roberta-base model for predicting 3 multiclass categories, with the class weights calculated above.
#use_cuda=True for running on GPU
#training for 4 epochs - try other values too!
model = ClassificationModel('roberta', 'roberta-base', num_labels=3, weight=class_weights.tolist(),
                            use_cuda=True, args={'reprocess_input_data': True, 'overwrite_output_dir': True,
                                                  "num_train_epochs": 10})

#train the model on the entire dataframe
model.train_model(df)

from pandas.core.groupby import DataFrameGroupBy
# create new dataframe with all rows including the Gold Standard
dfall = pd.read_excel('Classifications_Output.xlsx')
dfall

dfall.shape

dfall["Q2"] = dfall["Q2"]. replace(np. nan,0)
dfall["Q2"] = dfall["Q2"].replace(3,0)
dfall["Q2"] = dfall["Q2"].replace(4,0)

# rename columns
dfall["text"] = dfall["SNIPPET"]
dfall["labels"] = dfall["Q2"]

# Function to replace the token before the sentiment word with the Q1 info
def replace_token_with_number(row):
    string = row['SNIPPET']
    number = row['Q1']
    return string.replace('[[', "xxproj " + str(number) + " ")

# Apply the function to every row in the dataframe
dfall["text"] = dfall.apply(replace_token_with_number, axis=1)

# remove [+] and [-] from text column and replace with ++ and --
dfall["text"] = dfall["text"].str.replace("[+]", " xxpositive", regex=False)
dfall["text"] = dfall["text"].str.replace("[-]", " xxnegative", regex=False)

# remove [[ and  ]] from text column
dfall["text"] = dfall["text"].str.replace("[[", "", regex=False)
dfall["text"] = dfall["text"].str.replace("]]", "", regex=False)

set(dfall["labels"])

# predict Q2 values on entire dataframe including Gold Standard
predictionall = model.predict(dfall["text"].to_list())

#add new column to dataframe with roBERTa predictions
dfall['roBERTa'] = pd.Series(predictionall[0])

dfall.to_excel("Classifications_Output_v2.xlsx")