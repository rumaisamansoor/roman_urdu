#!/usr/bin/env python
# coding: utf-8

# In[1]:


import datetime
import pyodbc


start =  datetime.datetime.now()
conn= pyodbc.connect(
    "Driver=SQL Server Native Client 11.0;"
    "Server=DMK\SQLEXPRESS;"
    "Database=tempdb;"
    "Trusted_Connection=yes;"
    )


print("connection established")


# def read(conn):
   
#     list1=[]
#     print("in read")
#     cursor=conn.cursor()
#     cursor.execute('select * from sentiment_testdb')
  
#     for row in cursor:
        
#         list1.append(row)
#     print(list1)

# def update(conn):
#     check="checkkk"
#     id=2
#     cursor=conn.cursor()
#     cursor.execute("""update sentiment_testdb set sentiments =? where id = ?""",check, id)
#     conn.commit()
#     read(conn)
  

# read(conn)
# update(conn)
# print("done")


# In[ ]:



# In[3]:


import pandas as pd
SQL_Query = pd.read_sql_query('''select id,Comment,sentiment from sentiment_testdb''', conn)


# In[5]:


df=SQL_Query
df = df[df['sentiment'].isna()]
df.head()


# In[6]:


import tensorflow as tf
import keras
import numpy as np
import ast
from keras.preprocessing.sequence import pad_sequences
import pandas as pd

end = datetime.datetime.now()

# total time taken
time_diff = (end - start)
execution_time = time_diff.total_seconds() * 1000
print("library loading time: ",execution_time)


# In[7]:


start2 =  datetime.datetime.now()
file_content = open(r"C:\Users\Rumaisa\Desktop\sentiments\tokenizer_file.txt").read()
dicti = ast.literal_eval(file_content)

max_length=300


# In[55]:


new_model = tf.keras.models.load_model(r'C:\Users\Rumaisa\Desktop\sentiments\my_h5_model.h5')

test=df['Comment']
sentiment_score=[]
sentiment_labels=[]

for test_input in test:
    new_review = test_input
    new_review=str(new_review)
    new_review=new_review.lower()
    new_review= new_review.split()
    input_tokens=[]
    for i in new_review:
        try:
            input_tokens.append(dicti[i])
        except:
            input_tokens.append(0)

    input_tokens = list([input_tokens])
    
    #seq = tokenizer.texts_to_sequences(new_review)
    padded = pad_sequences(input_tokens, maxlen=max_length,padding='post')
    pred = new_model.predict(padded)
    labels = ['negative','positive']
    # print(pred, labels[np.argmax(pred)])
    temp=str(pred)+str(labels[np.argmax(pred)])
    sentiment_score.append(temp)
    pred=pred[0]
    x=pred[0]
    y=pred[1]
  
    if (x > 0.49 and x <0.53) or (y > 0.49 and y <0.53) :
        sentiment_labels.append("neutral")
    if (x> 0.53 and x< 0.75):
        sentiment_labels.append("negative")
    if (y> 0.53 and y< 0.75):
        sentiment_labels.append("positive")
    if (x > 0.75 and x <0.99):
        sentiment_labels.append("very negative")
    elif (y > 0.75 and y <0.99):
        sentiment_labels.append("very positive")


# In[56]:


import numpy

id_list=df['id'].to_numpy(dtype=numpy.int64)


# In[58]:


def read(conn):
   
    list1=[]
    print("in read")
    cursor=conn.cursor()
    cursor.execute('select * from sentiment_testdb')
  
    for row in cursor:
        
        list1.append(row)
    print(list1)

def update(conn):
    cursor=conn.cursor()
    for i in range(len(df['id'])):
        x=int(id_list[i])
       #sentiment[i]
        cursor.execute("""update sentiment_testdb set sentiment =? where id = ?""",sentiment_labels[i], x)
        conn.commit()
    
    read(conn)
  

update(conn)
print("done")


end2 = datetime.datetime.now()
time_diff2 = (end2 - start2)
execution_time2 = time_diff2.total_seconds() * 1000
print("code running time for 20 comments and saving to DB: ",execution_time2)

