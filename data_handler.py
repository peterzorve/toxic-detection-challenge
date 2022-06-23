

import pandas as pd 
import re 


data = pd.read_csv('data/train.csv')
# print(data.head(10))
# print(data.columns)
data = data.drop('id', axis=1)
# print(data)

def data_engineering(data):
     column_name, no, yes = [], [], []
     for i in data.columns:
          if i == 'comment_text':
               continue
          else:
               column_name.append(i),   no.append(data[i].value_counts()[0]),   yes.append(data[i].value_counts()[1])
     return column_name, no, yes

# column_name, no, yes = data_engineering(data)

# print(column_name)
# print(no)
# print(yes)

# print('BEFORE')
# print(data['comment_text'][3])

def preprocessing(data):
     # data = data 
     data['comment_text'] = data['comment_text'].apply(lambda text: text.lower()) 
     data['comment_text'] = data['comment_text'].apply(lambda text: text.strip()) 
     data['comment_text'] = data['comment_text'].apply(lambda text: re.sub('\n', ' ', text))
     data['comment_text'] = data['comment_text'].apply(lambda text: re.sub(',', ' ', text))
     data['comment_text'] = data['comment_text'].apply(lambda text: re.sub('.', ' ', text))
     data['comment_text'] = data['comment_text'].apply(lambda text: re.sub('\'', '', text))
     data['comment_text'] = data['comment_text'].apply(lambda text: re.sub('"', '', text))
     data['comment_text'] = data['comment_text'].apply(lambda text: re.sub('! ', '', text))
     data['comment_text'] = data['comment_text'].apply(lambda text: re.sub('/', '', text))
     data['comment_text'] = data['comment_text'].apply(lambda text: re.sub('-', ' ', text))
     data['comment_text'] = data['comment_text'].apply(lambda text: re.sub('\w*\d\w*\*', ' ', text))
     data['comment_text'] = data['comment_text'].apply(lambda text: re.sub(r'[^\x00-\x7f]', r' ', text)) 
     data['comment_text'] = data['comment_text'].apply(lambda text: text.strip())
     return data 

# print()
# print('AFTER')
data = (preprocessing(data))
# print(data['comment_text'][3])


def split_data(data):
     data_toxic          = data[['comment_text', 'toxic']]
     data_severe_toxic   = data[['comment_text', 'severe_toxic']]
     data_obscene        = data[['comment_text', 'obscene']]
     data_threat         = data[['comment_text', 'threat']]
     data_insult         = data[['comment_text', 'insult']]
     data_identity_hate  = data[['comment_text', 'identity_hate']]
     return data_toxic, data_severe_toxic, data_obscene, data_threat, data_insult, data_identity_hate 


data_toxic, data_severe_toxic, data_obscene, data_threat, data_insult, data_identity_hate = split_data(data)

print()
print(data_toxic.head(5))
print()
print(data_severe_toxic.head(5))
print()
print(data_obscene.head(5))
print()
print(data_threat.head(5))
print()
print(data_insult.head(5))
print()
print(data_identity_hate.head(5))
