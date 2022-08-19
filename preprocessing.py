import os
import pandas as pd


# MERGING CSV FILES OF EACH PAGE
# 1. defines path to csv files
path = "data/raws/"

# 2. creates list with csv files to merge based on name convention
file_list = [path + f for f in os.listdir(path)]

# 3. creates empty list to include the content of each file converted to pandas csv_merged
csv_list = []

# 4. reads each (sorted) file in file_list, converts it to pandas csv_merged and appends it to the csv_list
for file in sorted(file_list):
    csv_list.append(pd.read_csv(file).assign(File_Name = os.path.basename(file)))
    
# 5. merges single pandas csv_mergeds into a single csv_merged, index is refreshed 
csv_merged = pd.DataFrame(pd.concat(csv_list, ignore_index=True))



# PRE_PROCESSING OF DATASET
#drop useless columns
csv_merged.drop("File_Name", axis=1, inplace=True)
csv_merged.drop("Unnamed: 0", axis=1, inplace=True)

#delete duplicates:
csv_merged = csv_merged.drop_duplicates(keep=False)

#cleaning prices
csv_merged['prix'] = csv_merged['prix'].str[:-1]
csv_merged['prix']=csv_merged['prix'].str[:-1]
csv_merged['prix']=csv_merged['prix'].str.replace(',','.')
csv_merged['prix']=csv_merged['prix'].str.replace('\u202f','')
csv_merged['prix']=csv_merged['prix'].str.replace('No','0')
csv_merged['prix']=csv_merged['prix'].str.strip()
csv_merged["prix"] = pd.to_numeric(csv_merged["prix"])

#cleaning brand
csv_merged['marque']=csv_merged['marque'].str.strip()

#cleaning ratings
csv_merged['evaluations']=csv_merged['evaluations'].str.replace(',','.')
csv_merged['evaluations']=csv_merged['evaluations'].str.strip()
csv_merged["evaluations"] = pd.to_numeric(csv_merged["evaluations"])


#splliting reveiwer rating into 2 class
for i in csv_merged.index:
    if csv_merged["reviewer_evaluation"][i]=='1,0' or csv_merged["reviewer_evaluation"][i]=='2,0':
        csv_merged["reviewer_evaluation"][i]=0
    else:
        csv_merged["reviewer_evaluation"][i]=1

#convert reviews to lowercase
for i in csv_merged.index:
    csv_merged["avis_text"][i]=csv_merged["avis_text"][i].lower()

#Filtring brands + traiting Nan values 
for i in csv_merged.index:
    if 'IOS' in csv_merged["marque"][i] or 'Ios' in csv_merged["marque"][i]:
        csv_merged["marque"][i]='Apple'
    if 'Apple' in csv_merged["marque"][i] :
        csv_merged["marque"][i]='Apple'
    if 'grammes' in csv_merged["marque"][i] :
        csv_merged["marque"][i]='Samsung'
    if '10' in csv_merged["marque"][i]:
        csv_merged["marque"][i]='Android 10'
    if '11' in csv_merged["marque"][i]:
        csv_merged["marque"][i]='Android 11'
    if '12' in csv_merged["marque"][i]:
        csv_merged["marque"][i]='Android 12'
    if csv_merged["prix"][i]==0 or csv_merged["prix"][i]=='None'or csv_merged["marque"][i]=='None':
        csv_merged.drop(i, inplace=True)

#Translating 
from deep_translator import GoogleTranslator

#sentiment analysis
import tensorflow as tf
from transformers import DistilBertTokenizerFast
from transformers import TFDistilBertForSequenceClassification

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
loaded_model = TFDistilBertForSequenceClassification.from_pretrained("./sentiment")

def sentiment_mod(translated):
    predict_input = tokenizer.encode(translated,
                                    truncation=True,
                                    padding=True,
                                    return_tensors="tf")
    tf_output = loaded_model.predict(predict_input)[0]
    tf_prediction = tf.nn.softmax(tf_output, axis=1)
    label = tf.argmax(tf_prediction, axis=1)
    label = label.numpy()
    return(label[0])

for i in csv_merged.index:
    translated = GoogleTranslator(source='auto', target='english').translate(csv_merged["avis_text"][i][:4999]) 
    csv_merged.loc[i, 'sentiment'] = sentiment_mod(translated)
    if sentiment_mod(translated)!= csv_merged["reviewer_evaluation"][i]:
        csv_merged.loc[i, 'sentiment'] = csv_merged["reviewer_evaluation"][i]

#we no longer need this column  
csv_merged.drop("reviewer_evaluation", axis=1, inplace=True)

#saving file 
csv_merged.to_csv('data/dataset/data.csv', index=False)


