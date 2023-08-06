#!/usr/bin/env python
# coding: utf-8

# # 3.allow user to select reference corpus
# 
# #Feature3

# In[23]:


dataset1 = "Brown corpus"
dataset2 = "Enron corpus"
while 1:
    selected_dataset = input(f'Select Courpus you want to use: \n 1: {dataset1} 2: {dataset2}\n')
    selected_dataset = int(selected_dataset)
    if(selected_dataset == 1 or selected_dataset ==2):
        exec_command = f"print(f'SUCCESS: You chose: " + str(selected_dataset) + " " + "{dataset" + str(selected_dataset) + "}')"
        exec(exec_command)
        break
    print('Please input de cimal number\n')
    try:
        selected_dataset = int(selected_dataset)
    except:
        print('Please input decimal number\n')


# ## Remove NaN and Change Enron Documents into one Document

# In[24]:


import pandas as pd
import numpy as np
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
def rmNan(df):
    #remove nan
    for i, msg in enumerate(df['body']):
        # print(i, msg)
        if msg is np.nan:
            df = df.drop(i)
    for i, msg in enumerate(df['body']):
        if msg is np.nan:
            print(i, msg) 

    #merge documents into one
    str_all_document=''
    for index, record in df.iterrows():
        str_all_document = str_all_document + str(record[1])
    del df
    
    return pd.DataFrame({"author":["ENRON DATASET"],
                        "message": [str_all_document]})#rename culmun 'body' to 'message'
            

# ## Make DataFrame, Q, K1, K2, and Enron or Brown Datasets

# In[25]:


import pandas as pd
import numpy as np
import glob
def makeDataset(datasetnum = 0):
    data = ""
    if(datasetnum == 2):
        df = pd.read_csv("./data/preprocessed_enron.csv")
        df = rmNan(df)
    elif(datasetnum== 1):
        files = glob.glob("./data/Brown/*")
        for file in files:
            f = open(file, 'r')
            data = f.read() + ' ' + data
            f.close()
            break
        df = pd.DataFrame({"author":["BROWN DATASET"],
                            "message": [data]})
        print(data)
    return df.reset_index(drop=True)

# In[28]:


#Make df_Q dataset 
f = open('./data/Q_dataset.txt', 'r')
data = f.read()
f.close()
df_Q = pd.DataFrame({"author":["Q DATASET"],
                    "message": [data]})
                    

# In[29]:


#Make df_K1 dataset 
f = open('./data/K1_dataset.txt', 'r')
data = f.read()
f.close()
df_K1 = pd.DataFrame({"author":["K1 DATASET"],
                    "message": [data]})


# In[30]:


#Make df_K2 dataset 
f = open('./data/K2_dataset.txt', 'r')
data = f.read()
f.close()
df_K2 = pd.DataFrame({"author":["K2 DATASET"],
                    "message": [data]})

# In[31]:


#Make df_ref dataset 
df_ref = makeDataset(selected_dataset)

# In[32]:


df = pd.concat([df_Q,df_K1, df_K2, df_ref])
df = df.reset_index(drop=True)
del df_Q, df_K1,df_K2,df_ref

# In[33]:


df

# # 1. count, list and order the frequency of words 
# 
# #Feature1

# In[34]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
def tokenizeFunc(documents):
    # documents = df['message'].tolist()
    tf_vectorizer = CountVectorizer()
    tf_vectors = tf_vectorizer.fit_transform(documents)         # word frequency list
    return tf_vectors, tf_vectorizer
# for i, msg in enumerate(df['message']):
#     # print(i, msg)
#     if msg is np.nan:
#         print(i, msg)
#tfidf_vectorizer = TfidfVectorizer()
#tfidf_vectors = tfidf_vectorizer.fit_transform(documents) # keyword frequency list
# tf_vectorizer = CountVectorizer()
# tf_vectors = tf_vectorizer.fit_transform(documents)         # word frequency list
# del tf_vectorizer
# del tfidf_vectorizer# データ分割r

# ### Make All datasets a list and make tf vector and tf vectrizer

# In[35]:


documents= list(df['message'])
tf_vectors, tf_vectorizer = tokenizeFunc(documents)
documents


# In[36]:


#[remove]Dataframeや単語リストが一つのDFで十分な場合削除　7/25日米田

# tf_vectors_Q, tf_vectorizer_Q = tokenizeFunc(df_Q)
# tf_vectors_K2, tf_vectorizer_K1 = tokenizeFunc(df_K1)
# tf_vectors_K2, tf_vectorizer_K2 = tokenizeFunc(df_K2)
# tf_vectors_ref, tf_vectorizer_ref = tokenizeFunc(df_ref)

# words_Q=tf_vectorizer_Q.get_feature_names_out()
# words_K1=tf_vectorizer_K1.get_feature_names_out()
# words_K2=tf_vectorizer_K2.get_feature_names_out()
# words_K3=tf_vectorizer_ref.get_feature_names_out()

# ## Make a words dictionary in all documents 

# In[37]:


# 作成された辞書を作る　:トレインデータ・テストデータ両方に対応
words=tf_vectorizer.get_feature_names_out()

# ## Make the Words frequency matrix 

# ### This Matrix's row indices ared corresponding with a document in the 

# In[38]:


#First row: dataset Q
#Second row: dataset K1
#Third row: dataset k2
#Fourth row: dataset ref

# ### This Matrix's col indices are corresponding with a word in the above document

# In[39]:


words

# ## Insert Words Frequemcy Vector into 'tf' Column in DF for Each Dataset

# In[40]:


tf_mat = tf_vectors.toarray()
del tf_vectors
df['tf'] = tf_mat.tolist()

# In[41]:


df

# In[42]:


#[remove]ここの記述で必要な部分は上記に記述。そのた不要であれば削除

#tfidf_mat = tfidf_vectors.toarray() # dead every time
#del tfidf_vectors
# tf_mat = tf_vectors.toarray()
# del tf_vectors
# df['tf'] = tf_mat.tolist()
#df['tfidf'] = tfidf_mat.tolist()

# In[43]:


#[remove]今回はからのデータセットが存在しないため。確認後削除

# 0 ベクトルを消去 Normalization のため
# for i, vec in enumerate(df['tf']):
#     if sum(vec) == 0:
#         df = df.drop(i)

# df = df.reset_index(drop=True)

# # 5. display the first 20 words of each dataset 
# 
# #Feature5

# In[44]:


def dispAndMakeWordFreq(df, words, author = 0):
    data = df.loc[author]
    freq = data['tf']
    wf = pd.DataFrame({'words': words, 'frequency': freq})
    wordli = []
    freqli =[]
    wordindexli = []
    for key, data in wf.iterrows():
        if(int(data[1]) != 0):
            wordli.append(data[0])
            freqli.append(data[1])
            wordindexli.append(key)
        
    return pd.DataFrame({'words': wordli, 'frequency': freqli, 'wordIndex': wordindexli})
    


# ## Words Frequency of Dataset Q

# In[45]:


wf_list_Q = dispAndMakeWordFreq(df,words, author = 0)
wf_list_Q = wf_list_Q.sort_values('frequency', ascending=False)
wf_list_Q = wf_list_Q.reset_index(drop=True)
wf_list_Q.head(20)

# ## Words Frequency of Dataset K1

# In[46]:


wf_list_K1 = dispAndMakeWordFreq(df,words, author = 1)
wf_list_K1 = wf_list_K1.sort_values('frequency', ascending=False)
wf_list_K1 = wf_list_K1.reset_index(drop=True)
wf_list_K1.head(20)

# ## Words Frequency of Dataset K2

# In[47]:


wf_list_K2 = dispAndMakeWordFreq(df,words, author = 2)

wf_list_K2 = wf_list_K2.sort_values('frequency', ascending=False)
wf_list_K2 = wf_list_K2.reset_index(drop=True)
wf_list_K2.head(20)

# ## Words Frequency of Dataset ref

# In[48]:


wf_list_ref = dispAndMakeWordFreq(df,words, author = 3)
wf_list_ref = wf_list_ref.sort_values('frequency', ascending=False)
wf_list_ref = wf_list_ref.reset_index(drop=True)
wf_list_ref.head(20)

# ## count, list and order the frequency of keywords
# #Feature2

# ## Normalization of Word Frequencies to All Datasets and Add them into DF

# In[49]:


# we generaly name 'ntf' for normalized term frequency
normalized_tf_list = []
for row in df['tf']:
    num_words = sum(row)
    normalized_tf = []
    for x in row:
        normalized_tf.append(x/num_words)
    normalized_tf_list.append(normalized_tf)

df['ntf'] = normalized_tf_list

# #Tf-idf の代わりに利用する keyness を作る
# 
# ここでは　df['keyness'] を作成し追加したい

# ## using Log ratio 

# In[50]:


import math
# we generaly name 'ntf' for normalized term frequency
# First, create the shared normalized tf vector
# shared_ntf = None # shared ntf of all document
# matrix = []
# for row in df['ntf']:
#     matrix.append(row)
# np_matrix = np.array(matrix)
# mean_vector = np_matrix.mean(axis=0)
# shared_ntf = mean_vector.tolist()

def keyness(ntf_vector1, ref_ntf_vector2): # freq_vector1 and freq_vector2 are both already normalized
    keyness_vec = []
    for i, x in enumerate(ntf_vector1):
        if ntf_vector1[i] == 0 or ref_ntf_vector2[i] == 0:
            keyness_vec.append(0)
        else:
            keyness_vec.append(math.log2(ntf_vector1[i]/ref_ntf_vector2[i]))

    return keyness_vec


keyness_mat = []
for ntf_vector in df['ntf']:
    ntf_ref = df['ntf'][3]
    keyness_vec = keyness(ntf_vector, ntf_ref)
    keyness_mat.append(keyness_vec)

# keyness を　Dataframe に追加
df['keyness'] = keyness_mat

# In[51]:


df

# # Display the first 20 keywords of each dataset 

# In[52]:


def dispAndMakeKeyWordList(df, words, author = 0):
    data = df.loc[author]
    keyness = data['keyness']
    wf = pd.DataFrame({'words': words, 'keyness': keyness})
    wordli = []
    freqli =[]
    wordindexli = []
    for key, data in wf.iterrows():
        if(int(data[1]) != 0):
            wordli.append(data[0])
            freqli.append(data[1])
            wordindexli.append(key)
        
    return pd.DataFrame({'words': wordli, 'keyness': freqli, 'wordIndex': wordindexli})

# ## Keyword of Dataset Q

# In[53]:


keyword_list_Q = dispAndMakeKeyWordList(df,words, author = 0)

keyword_list_Q = keyword_list_Q.sort_values('keyness',ascending=False)
keyword_list_Q = keyword_list_Q.reset_index(drop=True)
keyword_list_Q.head(20)

# ## Keyword of Dataset K1

# In[54]:


keyword_list_K1 = dispAndMakeKeyWordList(df,words, author = 1)
keyword_list_K1 = keyword_list_K1.sort_values('keyness',ascending=False)
keyword_list_K1 = keyword_list_K1.reset_index(drop=True)
keyword_list_K1.head(20)

# ## Keyword of Dataset K2

# In[55]:


keyword_list_K2 = dispAndMakeKeyWordList(df,words, author = 2)

keyword_list_K2 = keyword_list_K2.sort_values('keyness',ascending=False)
keyword_list_K2 = keyword_list_K2.reset_index(drop=True)
keyword_list_K2.head(20)

# # 7. display the shared words in the first 20 words of each dataset
# #Feature7

# In[56]:


def dispAndMakeSharedWordsFreq(df,df_ref):
    df = df[df['words'].isin(df_ref['words'])] #filtering with the words in df2
    df_ref = df_ref[df_ref['words'].isin(df['words'])] #filtering with the words in df
    #now the words in df and df2 are same
    #sort words in the alphabetical order to become the same words as the same rows
    df = df.sort_values('words')
    df_ref = df_ref.sort_values('words')
    #merge df2 frequency to df1 
    df['ref_frequency'] = list(df_ref['frequency'])
    df['shared_word_keyword_frequency'] = (df['frequency'] + df['ref_frequency'])
    return df

# ## Shared Words Frequency in Dataset Q and K1

# In[57]:


#SWF = Shared Word Frequency
SWF_QandK1 = dispAndMakeSharedWordsFreq(wf_list_Q,wf_list_K1)
SWF_QandK1=SWF_QandK1.sort_values('frequency', ascending=False)
SWF_QandK1 = SWF_QandK1.reset_index(drop=True)
SWF_QandK1.head(20)


# ## Shared Words Frequency in Dataset Q and K2

# In[58]:


#SWF = Shared Word Frequency
SWF_QandK2 = dispAndMakeSharedWordsFreq(wf_list_Q,wf_list_K2)
SWF_QandK2 = SWF_QandK2.sort_values('frequency', ascending=False)
SWF_QandK2 = SWF_QandK2.reset_index(drop=True)
SWF_QandK2.head(20)

# ## Display the shared keywords in the first 20 keywords of each dataset

# In[59]:


def dispAndMakeSharedKeyword(df,df_ref):
    df = df[df['words'].isin(df_ref['words'])] #filtering with the words in df2
    df_ref = df_ref[df_ref['words'].isin(df['words'])] #filtering with the words in df
    #now the words in df and df2 are same
    #sort words in the alphabetical order to become the same words as the same rows
    df = df.sort_values('words')
    df_ref = df_ref.sort_values('words')
    #merge df2 frequency to df1 
    df['ref_keyness'] = list(df_ref['keyness'])
    df['wordIndex'] = list(df['wordIndex'])
    
    return df

# ## Shared Keywords in Dataset Q and K1

# In[60]:


SK_QandK1 = dispAndMakeSharedKeyword(keyword_list_Q,keyword_list_K1)
SK_QandK1 = SK_QandK1.sort_values('keyness', ascending=False)
SK_QandK1 = SK_QandK1.reset_index(drop=True)
SK_QandK1.head(20)

# ## Shared Keywords in Dataset Q and K2

# In[61]:


SK_QandK2 = dispAndMakeSharedKeyword(keyword_list_Q,keyword_list_K2)
SK_QandK2 = SK_QandK2.sort_values('keyness', ascending=False)
SK_QandK2 = SK_QandK2.reset_index(drop=True)
SK_QandK2.head(20)

# # Remake Keyness values and Shared Words Frequency for Each Author 

# In[62]:


#Only filterling the keyness and SFW values using sheared word freq and shared keyword.
#this objective is to adjust the length of shared keyword and word list to original size of 'words'
def remakeKeynessAndFW(df, words,new_SWF,new_keyness, authorId = 0 ):
    if(authorId==0):
        print('dataset Q is not allowed')
        return [],[]
    #init list by 0
    keynessli=[]
    SFWli = []
    for k in range(len(words)):
        keynessli.append(0)
        SFWli.append(0)
    #end init
    #make new shared word frequency list
    #items[] compounds of: words,	frequency,	wordIndex,	ref_frequency
    for key,items in new_SWF.iterrows():
        SFWli[items[2]] = items[3]
    #make new shared keyword list
    # items[] compounds of: words,	keyness,	wordIndex,	ref_keyness
    for key, items in new_keyness.iterrows():
        keynessli[items[2]] = items[3]
    return SFWli, keynessli

# ### Now K1 and K2 in df have the tf and keyness based on the existence of each shared word freq. and shared keywords as new datasets

# In[63]:


df['tf'][1], df['keyness'][1] = remakeKeynessAndFW(df, words, SWF_QandK1, SK_QandK1, 1)
df['tf'][2], df['keyness'][2] = remakeKeynessAndFW(df, words, SWF_QandK2, SK_QandK2, 2)

# In[64]:


df

# # Prediction

# In[65]:


# # start からend までのwordの配列を返す
# def extract_features_words(freq_vector, words, start=0, end=20):
#     setX = set(freq_vector[0]) # 最大値を取り出すため set を作成
#     count = 0
#     result = []
#     while count<end:
#         max_value = max(setX)
#         max_index = freq_vector.index(max_value)
#         max_word = words[max_index]
#         setX.remove(max_value)
#         ### if exclude stopwords
#         # if max_word not in stop_words:
#         #     if count>= start:
#         #         result.append(max_index)
#         #     count+=1
#         if count>= start:
#             result.append(max_word)
#         count += 1
#     return result

# In[66]:


# start からend までのword IDの配列を返す
def extract_features(freq_vector, words, start=0, end=20):
    freq_vector = freq_vector[0].copy()
    setX = freq_vector # 最大値を取り出すため set を作成
    count = 0
    result=[]
    while count<end:
        try:
            max_value = max(setX)
            # print(max_value)
        except ValueError:
            print('valueerror')
            return result
        max_index = freq_vector.index(max_value)
        max_word = words[max_index]
        setX.remove(max_value)
        if count>= start:
            result.append(max_word)
        count += 1
        ### if exclude stopwords
        # if max_word not in stop_words:
        #     if count>= start:
        #         result.append(max_index)
        #     count+=1

        
    return result

#testcase

# extract_features(df['tf'].tolist(), words, 0 , 20)

# In[67]:


def get_similarity(feature_vector1,feature_vector2):
    return len(set(feature_vector1) & set(feature_vector2))

# In[68]:


INF = float('inf')

def predict(questioned_vector,candidates_vectors):
    #initialize-------------------------------------
    start = 0
    end = 20
    similarityWithQ_tf = {}
    similarityWithQ_keyness = {}
    suspected = list(candidates_vectors['author'])
    #prepare questioned tf and keyword features
    
    while(len(suspected) > 1):
        Q_features_tf = extract_features(questioned_vector['tf'].tolist(), words, start, end)
        Q_features_keyness = extract_features(questioned_vector['keyness'].tolist(), words, start, end)
        for idx, candidates_items in candidates_vectors.iterrows():
            author = candidates_items[0]
            
            #prepare candidates tf and keynes features
            candidates_vector_tf = [candidates_items[2]] 
            candidates_vector_keyness = [candidates_items[4]]
            if author in suspected:
                print('Analysed Author Information')
                #tf-------------------
                C_features_tf = extract_features(candidates_vector_tf, words, start, end)
                score_tf = get_similarity(C_features_tf,Q_features_tf)
                similarityWithQ_tf[author]=score_tf
                print(f'{author}\'s similality tf score = {score_tf}')
                
                #keyness----------------
                C_features_keyness = extract_features(candidates_vector_keyness, words, start, end)
                score_keyness = get_similarity(C_features_keyness,Q_features_keyness)
                similarityWithQ_keyness[author]=score_keyness
                print(f'{author}\'s similality keyness score = {score_keyness}')
                print('...')
        innocent = min(similarityWithQ_keyness, key=similarityWithQ_keyness.get)
        suspect =  max(similarityWithQ_tf, key=similarityWithQ_tf.get)

        #asking what they want to do
        act = 0
        while(1):
            if(len(suspected) == 1):
                print('*********************************')
                print(f'Final Result of the suspectful auther is: {suspect}')
                print('Thank you.')
                print('*********************************')
                return suspect
            print('*********************************')
            print(f'The MOST suspectful auther based on shared keyword frequency: {suspect}')
            print(f'The LEAST suspectful auther based on Keyness: {innocent}')
            print('*********************************')
            act = input(f'Do you wan to remove the LEAST suspectful one from searching, \"{innocent}\"?(yes: 1, no: 0)')
            try:
                if(int(act) == 1):
                    print('...')
                    print('remove the user from candidates')
                    suspected.remove(innocent)
                else:
                    print('...')
                    print('Go to next 20 words searching')
                    break
            except:
                print('Please input decimal number\n')
                break
        end += 20

    return suspected[0]

# ### Divided df into Q and Candidates

# In[69]:


questioned_df = df[0:1].copy()
references_df = df[1:3].copy()
references_df.reset_index
questioned_df

# In[70]:


references_df

# In[71]:


suspect = predict(questioned_df, references_df)
# suspect = predict(questioned_df, df)


# # Garbege codes

# ## Check function in Train Data

# In[72]:


i = 0
bad_guy = predict(df['keyness'][i], reference_vectors)
print(f'bad guy : {bad_guy}')
print('-'*20)
print(f"True author : {df['author'][i]}")

# In[ ]:


# all_test_data = len(X_keyness_test)# 

# In[ ]:


# match_cnt = 0
# all_test_data = len(X_keyness_train)
# for i in X_keyness_train.index:
#     bad_guy = predict(df['keyness'][i], reference_vectors)
#     if df['author'][i] == bad_guy:
#         match_cnt = match_cnt + 1

# print(f'Math rate is: {match_cnt/all_test_data*100} % ')
#     #print(f'bad guy : {bad_guy}')
#     #print('-'*20)
#     #print(f"True author : {df['author'][i]}")

# In[ ]:


# def predict(questioned_vector, reference_vectors):
#     suspected = [author for author in authors]

#     comparedSize = 20
#     while(len(suspected) > 1):

# In[ ]:


# def dispAndMakeSharedKeywordFreq(df,df_ref):
#     df = df[df['words'].isin(df_ref['words'])] #filtering with the words in df2
#     df_ref = df_ref[df_ref['words'].isin(df['words'])] #filtering with the words in df
#     #now the words in df and df2 are same
#     #sort words in the alphabetical order to become the same words as the same rows
#     df = df.sort_values('words')
#     df_ref = df_ref.sort_values('words')
#     #merge df2 frequency to df1 
#     df['ref_frequency'] = list(df_ref['frequency'])
#     df['shared_word_keyword_frequency'] = (df['frequency'] + df['ref_frequency'])
#     return df
## Keyword Frequency in Dataset Q and Ref
#KF = Keyword Frequency
# KF_QandRef = dispAndMakeSharedKeywordFreq(wf_list_Q,wf_list_ref)
# KF_QandRef.sort_values('frequency', ascending=False).head(20)
# ## Keyword Frequency in Dataset K1 and Ref
# #KF = Shared Keyword Frequency
# KF_QandRef = dispAndMakeSharedKeywordFreq(wf_list_K1,wf_list_ref)
# KF_QandRef.sort_values('frequency', ascending=False).head(20)
# ## Keyword Frequency in Dataset K2 and Ref
# #KF = Shared Keyword Frequency
# KF_QandRef = dispAndMakeSharedKeywordFreq(wf_list_K2,wf_list_ref)
# KF_QandRef.sort_values('frequency', ascending=False).head(20)

# In[ ]:


# import nltk
# from nltk.corpus import stopwords

# nltk.download('stopwords')
# stop_words = stopwords.words('english')

# In[ ]:


# #確認用
# i = 22
# msg = df['message'][i]
# max_value = max(keyness_mat[i])
# max_idx = keyness_mat[i].index(max_value)
# print(words[max_idx])

# print(msg)
# #print(df['author'][i])


# In[ ]:


# from sklearn.model_selection import train_test_split
# X_keyness_train, X_keyness_test, Y_keyness_train, Y_keyness_test = train_test_split(df['keyness'],df['author'],test_size=0.2,shuffle=True)
# X_tf_train, X_tf_test, Y_tf_train, Y_tf_test = train_test_split(df['tf'],df['author'],test_size=0.2,shuffle=True)

# In[ ]:


# # How many author?
# authors = set(Y_keyness_test)
# authors_list = [author for author in authors]

# # Create Reference_vectors
# size: 著者の数
# 
# Train データから作る

# In[ ]:


# # df_X = pd.DataFrame(X_tf_train.values.tolist())
# # df_Y = pd.DataFrame(Y_tf_train.values.tolist())


# df_train = pd.concat((X_keyness_train, Y_keyness_train.rename('author')), axis=1)
# #
# reference_vectors = {}
# for author in authors:

#     df_author = df_train.groupby('author').get_group(author)

#     matrix = []
#     for row in df_author['keyness']:
#         matrix.append(row)

#     np_matrix = np.array(matrix)

#     mean_vector = np_matrix.mean(axis=0)
#     reference_vectors[author] = mean_vector.tolist()

# In[ ]:


# for author, ref_vec in reference_vectors.items():
#     max_value = max(ref_vec)
#     max_idx =ref_vec.index(max_value)
#     print(f'{author}: {words[max_idx]}')


# In[ ]:


# authors
