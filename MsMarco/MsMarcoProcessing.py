import json
import pandas as pd
#define path for training and evaluation data
train_path = ''
val_path = ''
#MsMarco is a dictionary with keys: ['answers', 'passages', 'query', 'query_id', 'query_type', 'wellFormedAnswers']
#For our purposes we are only concerned with query, answers, and passages. However, not all questions in MsMarco have answers. Furthermore, the passages key for each questions has multiple passages, with only one being the correct passage that answers the question.
#In order to only select the questions that have answers and a correct passage we need to loop through the dictionary and make sure an answer is present. 
def ms_processing(path):
  with open(path) as f:
    data = json.load(f)
  questions = []
  context = []
  answers = []
  for x in range(len(data['query'])):
     x = str(x)

     if 'No Answer Present.' in data['answers'][x][0]:
          continue
     else:
         for a in data['passages'][x]:
             if a['is_selected']==1:
                 context.append(a['passage_text'])
                 questions.append(data['query'][x])
                 answers.append(data['answers'][x])
                 break
            
  return questions,context,answers
 
