import pandas as pd
import os
import re
import codecs
import hazm
from hazm import Normalizer , word_tokenize
from pandarallel import pandarallel
import argparse
from parsinorm import General_normalization , Special_numbers ,Date_time_to_text#,Tokenizer
from transformers import AutoConfig, AutoTokenizer, AutoModel
import parsinorm
from parsinorm import Abbreviation
import json
# from parsivar import POSTagger
import parsivar
# from parsivar import Tokenizer
from tqdm import tqdm
import math
import sys
import numpy as np
import csv


# tagger=hazm.POSTagger(model='pos_tagger.model')

with open('data/authors_names.txt', 'r') as file:
    list_authors=file.read().split('\n')


tagger=parsivar.POSTagger()
nmz = Normalizer()
persian_stopwords=set([nmz.normalize(w) for w in codecs.open('./functions/persian_stopwords', encoding='utf-8').read().split('\n') if w])
title_stopwords=set([nmz.normalize(w) for w in codecs.open('./functions/persian_stopwords', encoding='utf-8').read().split('\n') if w]+['و']+\
    ['کارشناسی', 'اظهارنظر', 'طرح', 'بررسی', 'کشور', 'طرح','مجلس'
       'بودجه', 'سال', 'ایران', 'شورای', 'اصلاح', 'اسلامی', 'توسعه'])
# body_stopwords=['فهرست مطالبفهرست جداول و نمودارچکیده','مطالبفهرست','فهرست مطالب','فهرست جداول','به نام خدافهرست مطالبفهرست جداول و نمودارچکیده','به نام خدا فهرست مطالبفهرست جداول و چکیده','به نام خدا'
#   'نمودارچکیده','نمودار',]
body_stopwords= ["به نام خدا", "فهرست", "مطالب", "مقدمه",'چکیده','جداول','تصاویر','نمودار','جدول','چکیده']
filter_words = ['تدوین:', 'نویسنده:', 'تهیه و تدوین:', 'ناظر علمی:', 'مدیر مطالعه:','تهیه و تدوین:']


def remove_stopwords(text,stopwords=persian_stopwords):
    if text=='' or text ==None:
      return text
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if not word in stopwords]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


def filter_sentence(text,phrases_to_exclude=filter_words):
    '''delete sentece that contians filter words from text'''
    sentences = text.split('.')

    # Initialize a list to store the filtered sentences
    filtered_sentences = []

    # Iterate over the sentences
    for sentence in sentences:
        # Check if the sentence contains any of the phrases to exclude
        if not any(phrase in sentence for phrase in phrases_to_exclude):
            filtered_sentences.append(sentence)

    # Merge the filtered sentences back into a single text
    merged_text = '.'.join(filtered_sentences)

    return merged_text
    
def remove_stopwords_combinations(text,stopwords=body_stopwords):
    clean_text=[]
    for sentence in text.split('.')[:5]:
        # Check if any combination of words appears in the sentence
        combination_count = 0
        for word in stopwords:
            if word in sentence:
                combination_count += 1

        # Delete combinations of words from the sentence
        if combination_count > 1:
            for word in stopwords:
                sentence = sentence.replace(word, "")
        clean_text.append(sentence)
    
    clean_text+=text.split('.')[5:]
    clean_text='.'.join(clean_text)

    return clean_text.strip()



def fix_stopwords_spaces(text):
    patterns =[ r'به نام خدا(?! ی| وند)',r'مقدمه(?! ی| ای| های|ها|‌ی| ی| ای |‌ای)',r'چکیده(?! ی| ای| های|ها|‌ی| ی| ای |‌ای)',]
    stopwords=['به نام خدا', 'مقدمه','چکیده',]
    for pat ,word in zip(patterns,stopwords):
        text=re.sub(pat, f' {word} ', text)    
    
    return text

    
def drop_non_verb_sent(text,sent_id,tagger):
    # tagged=tagger.tag(word_tokenize(text))
    # senteces=text.split('.')
    filtered_ids = []
    for id in sent_id:
        sent=text[i]
        tagged_words=tagger.tag(word_tokenize(sent))
        if any(tag.startswith('V') for _, tag in tagged_words):
            filtered_ids.append(id)


    return filtered_ids

def replace_sequence_of_dots(text):
    # Define the regular expression pattern to match four or more dots
    pattern = r'\.{5,}'

    # Replace the matched pattern with a space
    text = re.sub(pattern, ' ', text)
    text=text.replace('....',' غیره ')
    text=text.replace('...',' غیره ')
    text=text.replace('..',' ')

    return text   

def delete_authors_names(text,authors_names=list_authors):
    
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if not word in authors_names]
    filtered_text = ' '.join(filtered_tokens)
    
    return filtered_text
    
def handle_dot_in_number(text):
    text = text.replace('۰', '0').replace('۱', '1').replace('۲', '2').replace('۳', '3')\
        .replace('۴', '4').replace('۵', '5').replace('۶', '6').replace('۷', '7').replace('۸', '8').replace('۹', '9')
# Replace dots between numbers with backslashes
    text = re.sub(r'(\d+)\.(\d+)', r'\1/\2', text)
    # Replace dot after numbers with hyphen
    text = re.sub(r'(\d+)\.', r'\1-', text)
    return text

def normalize_farsi(sent):

  if  str(sent)=='nan':
    return ''
  normalizer = General_normalization()
  special_numbers = Special_numbers()
  abbreviation = Abbreviation()
  date_time_to_text = Date_time_to_text()


  tokenizer = parsinorm.Tokenizer()
  
  sent=replace_sequence_of_dots(sent)
  sent=handle_dot_in_number(sent)
  sent=date_time_to_text.date_to_text(sent)
  sent=date_time_to_text.time_to_text(sent)

  sent=normalizer.punctuation_correction(sent)
  sent=filter_sentence(sent)  
  sent=normalizer.alphabet_correction(sent)
  sent=normalizer.semi_space_correction(sent)
  sent=normalizer.english_correction(sent)
  sent=normalizer.html_correction(sent)
  sent=normalizer.arabic_correction(sent)
  sent=normalizer.specials_chars(sent)
  sent=normalizer.remove_emojis(sent)
  sent=normalizer.unique_floating_point(sent)
  sent=normalizer.remove_comma_between_numbers(sent)
  sent=normalizer.number_correction(sent)
  sent=normalizer.remove_repeated_punctuation(sent)
  sent=abbreviation.replace_date_abbreviation(sent)
  sent=abbreviation.replace_persian_label_abbreviation(sent)
  sent=abbreviation.replace_law_abbreviation(sent)
  sent=abbreviation.replace_book_abbreviation(sent)
  sent=abbreviation.replace_other_abbreviation(sent)
  sent=abbreviation.replace_English_abbrevations(sent)
  

  sent=fix_stopwords_spaces(sent)
  sent=remove_stopwords_combinations(sent)
  sent=delete_authors_names(sent)
  sent=handle_dot_in_number(sent)
  
  return sent



def process_keywords(k_word):
    if bool(k_word):
        # print(keyword)
        keywords=list(k_word.values())[0]
        keywords=' '.join(keywords)
        keywords=re.sub(r'[^\w\s]', '', keywords)
        keywords=re.sub("\d+ ", " ", keywords)
        keywords=remove_stopwords(keywords)


        return normalize_farsi(keywords)
    else:
        return None


def process_title(title):
    title=re.sub(r'[^\w\s]', '', title)
    title=re.sub("\d+ ", " ", title)
    title=remove_stopwords(title)
    return normalize_farsi(title)


def process_summary(summary):
    summary=summary.replace('خلاصه:','')
    summary=summary.replace('خلاصه :','')

    return normalize_farsi(summary)
    
def merge_data(splited_df,return_last_part=False):
    gk=splited_df.groupby('file_id')
    new_rows=[]

    for file_id in splited_df.file_id.unique():
        group=gk.get_group(file_id)

        gourp=group.sort_values(by=['file_part'])
        if return_last_part:
            group=group.tail(2)

        n_sent=0
        merged_sent_ids=[]
        merged_sent_score=[]
        merged_text=[]
        for _,row in group.iterrows():
            merged_sent_ids.extend(row['sent_id']+n_sent)
            merged_sent_score.extend(row['sent_score'][0])
            merged_text.append(row['text'])       
            n_sent+=row['n_sent']
        sorted_candidate=sorted(list(zip(merged_sent_score,merged_sent_ids)),key=lambda x:x[0])

        merged_row={
            # 'text':tokenizer.sentence_tokenize('.'.join(merged_text),verb_seperator=False),
            'text':'.'.join(merged_text).split("."),
        'summary':row['summary'].split("<q>"),
        'sent_id': list(list(zip(*sorted_candidate))[1]) ,# get list of  second elements of tuples
        # 'sorted_candidate':sorted_candidate,
        'title':remove_stopwords(row['title'][0]),
        'keywords':remove_stopwords(row['keywords'][0]),
        'sent_score':merged_sent_score,'n_sent':n_sent,'file_id':file_id}
        new_rows.append(merged_row)
        
            
    new_df=pd.DataFrame(new_rows)
    return new_df

def process_single_row(df):
    new_rows=[]
    for _,row in df.iterrows():
        merged_row={
            # 'text':tokenizer.sentence_tokenize('.'.join(merged_text),verb_seperator=False),
            'text':row['text'].split("."),
        'summary':row['summary'].split("<q>"),
        'sent_id':row['sent_id'][0] ,# get list of  second elements of tuples
        # 'sorted_candidate':sorted_candidate,
        'title':remove_stopwords(row['title'][0]),
        'keywords':remove_stopwords(row['keywords'][0]),
        'sent_score':row['sent_score'][0],'n_sent':len(row['text'].split("."))
        ,'file_id':0}
        new_rows.append(merged_row)    
    new_df=pd.DataFrame(new_rows)
    return new_df



# csv.field_size_limit(sys.maxsize)

my_tokenizer = parsivar.Tokenizer()
tagger = parsivar.POSTagger()


def pos(text):
  word_with_tag = []
  try:
      sentences = my_tokenizer.tokenize_sentences(text)
      for s in sentences:
          tags = tagger.parse(my_tokenizer.tokenize_words(s))
          temp = [t[0]+'/'+t[1] for t in tags]
          word_with_tag.append(temp[:-1])
  except:
      word_with_tag = []
  return word_with_tag

def convert_to_json(raw_data, dst_path):
  os.makedirs(dst_path,exist_ok=True)
  tqdm.pandas(desc="tgt..")
  raw_data['tgt'] = raw_data['summary'].progress_apply(pos)
  tqdm.pandas(desc="src..")
  raw_data['src'] = raw_data['text'].progress_apply(pos)
  for idx, row in raw_data.iterrows():
      # if row['src'] == [] :#or row['tgt'] == []:
      #     continue
      # else:
      mydict = {}
      temp = []
      mydict['src'] = row['src']
      mydict['tgt'] = row['tgt']
      mydict['title']=row['Title']
      mydict['keywords']=row['Keywords']
      temp.append(mydict)
      try:
        filename = 'd'+ str(row['id']) + '.json'
      except:
        filename = 'd'+ str(idx) + '.json'

      # print(os.path.join(dst_path, filename))
      with open(os.path.join(dst_path, filename), "w", encoding='utf8') as json_file:
          json.dump(temp, json_file, ensure_ascii=False)


def split_long_data(df):
  '''
  df:dataframe with text summary Title, Keywords coulums with one row
  return dataframe with multiplue rows
  '''
  # split text to 510 words chunks
  new_rows=[]

  tokenizer = parsinorm.Tokenizer()
  for idx,row in df.iterrows():
      text=row["text"]
      # sentences=tokenizer.sentence_tokenize(text,verb_seperator=False)
      sentences=text.split('.')
      text_size=len(tokenizer.word_tokenize(text))
      words=tokenizer.word_tokenize(text)
      cum_sentences=np.cumsum([len(tokenizer.word_tokenize(sent)) for sent in sentences])
      split_point=[0]
      for i,cum in enumerate(cum_sentences):
          if cum >510*(len(split_point)):
              split_point.append(i-1)
      split_point.append(len(cum_sentences))
      # chunks = []
      for i in range(1,len(split_point)):
          chunk='.'.join(sentences[split_point[i-1]:split_point[i]])
          n_row={'id':f'{idx}-{i}','text':chunk,'summary':row["summary"],"Title":row['Title'],"Keywords":row['Keywords']}
          new_rows.append(n_row)
  new_df=pd.DataFrame(new_rows)
  return new_df




def keyword_check(text,keywords):
    flag=0
    # print(text)
    if keywords!=None:
        for k_word in keywords:
            for idx,sent in enumerate(text):
                if k_word in sent:
                    flag+=1
                    # print(' '.join(keywords)in sent)
                    # print(idx)
                    break
    return flag


def get_index_sent_with_kword(text,keywords):
    sent_index=[]
    # print(text)
    if keywords!=None:
        for k_word in keywords:
            for idx,sent in enumerate(text):
                if k_word in sent:
                    sent_index.append(idx)
    return sent_index




def get_commen_id(sent_id,k_word_id,title_id,conclusion_id):
    union_list = []

    union_list.extend(k_word_id)
    union_list.extend(title_id)
    union_list.extend(conclusion_id)
    union_list=list(set(union_list))
    #[(x,i) for i,x in enumerate(sent_id) if x in union_list]
     
    return [x for x in sent_id if x in union_list]


def extract_summary(text,sent_id):
    summary=[]
    for id in sent_id:
        summary.append(text[id])
    return '. '.join(summary[:10])


def create_idx_real_summary(df,summary_idx_path,n_sent=10):
    data = []
    with open(summary_idx_path, "a") as json_file:
        for _,row in df.iterrows():
            # print(row['sent_id'][:n_sent])
            json.dump({"sent_id":list(map(int,row['sent_id'][:n_sent]))}, json_file)
            json_file.write("\n")


def create_idx_keyword_summary(df,summary_idx_path,n_sent=10):
    data = []
    with open(summary_idx_path, "a") as json_file:
        for _,row in df.iterrows():
                
            if len(row['keyword_summary'])>=n_sent:
                sent_id=row['keyword_summary'][:n_sent]
 
            else:
                sent_id=row['keyword_summary']+row['sent_id']
            json.dump({"sent_id":list(map(int,sent_id))}, json_file)
            json_file.write("\n")

            


def create_matchsum_inputs(df,file_name,save_to,n_sent=10):
    '''
    file_name=string that names of created files start with it
    save_to=path for saving files
    n_sent= number of candidates
    
    '''
    os.makedirs(save_to,exist_ok=True)
    idx_r_path=os.path.join(save_to,f"{file_name}_idx_real_summary.jsonl")
    idx_k_path=os.path.join(save_to,f"{file_name}_idx_keyword_summary.jsonl")
    data_path=os.path.join(save_to,f'{file_name}_data.jsonl')
    
    if os.path.exists(data_path):
        os.remove(idx_k_path)
        os.remove(idx_r_path)
        os.remove(data_path)

    create_idx_real_summary(df,idx_r_path,n_sent)
    create_idx_keyword_summary(df,idx_k_path,n_sent)
    df.to_json(data_path,lines=True, orient="records",force_ascii=False)


    return idx_r_path,idx_k_path,data_path





def delete_folder_contents(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)


def extract_fetures(df):
    df['n_ext_summary']=df.apply(lambda row: len(row['sent_id']),axis=1)
    conclusion_keywords=['جمع بندی','نتایج','نتیجه گیری','جمع‌بندی','نتیجه گیری','ونتیجه‌گیری','وجمع‌بندی']

    df['have_conclusion_keyword']=df.text.apply(keyword_check,keywords=conclusion_keywords)
    df['indx_conclusion_keyword']=df.text.apply(get_index_sent_with_kword,keywords=conclusion_keywords)


    df['title_list']=df['title'].apply(lambda x: x.split(' ')if x!=None else None)
    df['have_title']=df.apply(lambda row: keyword_check(row['text'],row["title_list"]),axis=1)
    df['idx_title']=df.apply(lambda row: get_index_sent_with_kword(row['text'],row["title_list"]),axis=1)

    df['keywords_list']=df['keywords'].apply(lambda x: x.split(' ')if x!=None else None)
    df['have_keywords']=df.apply(lambda row: keyword_check(row['text'],row["keywords_list"]),axis=1)
    df['idx_keywords']=df.apply(lambda row: get_index_sent_with_kword(row['text'],row["keywords_list"]),axis=1)

    df['first_sent_summary']=df.apply(lambda row:row['text'][row['sent_id'][0]] ,axis=1)
    df["keyword_summary"]=df.apply(lambda row :get_commen_id(row['sent_id'],row["idx_keywords"],row["idx_title"],\
        row["indx_conclusion_keyword"]),axis=1)

    df['summary']=df.summary.apply(lambda y: y[0] if y==[''] else y)
    return df
