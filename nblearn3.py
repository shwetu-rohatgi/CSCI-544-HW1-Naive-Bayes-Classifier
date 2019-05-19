import sys
import glob
import os
import collections
import re
import numpy as np
import json

all_files = glob.glob(os.path.join(sys.argv[1], '*/*/*/*.txt'))

train_by_class = collections.defaultdict(list)

for f in all_files:
  class1, class2, fold, fname = f.split('/')[-4:]
  train_by_class[class1+class2].append(f)
    
#print (test_by_class)
#print('\n\n *** Test data:')
#print(json.dumps(test_by_class, indent=2))
#print('\n\n *** Train data:')
#print(json.dumps(train_by_class, indent=2))

vocab = {}

positive_deceptive, positive_truthful, negative_deceptive, negative_truthful = {},{},{},{}
def pre_processing(text_line, cls):
    text_line = text_line.replace('\n','')
    text_line = text_line.replace('\t','')
    #text_line = text_line.lower()
    
    cleaned_line = re.sub('[^a-z\s]+',' ',text_line)
    
    cleaned_line = re.sub('(\s+)',' ',cleaned_line)
    
    stop_words = ['1','2','3','4','5','6','7','8','9','0','it', 'hers', 'between', 'yourself', 'but', 'the','again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are','his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'up', 'to', 'ours', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than','hotel','stay','i','we', 'these', 'your','while', 'above', 'both','where', 'too', 'only','had', 'she', 'all','do', 'its', 'yours', 'such','chicago','day','ourselves','no', 'when', 'at', 'any','who', 'as', 'from']
    
    final_string = ""
    w_count = 0
    for word in cleaned_line.split():        
        if word in stop_words:
            continue
        else:
            final_string = final_string+" "+word
            w_count+=1
            if word in vocab:
                vocab[word]+=1
            else:
                vocab[word] = 1
            if cls=='positive_deceptive':
                if word in positive_deceptive:
                    positive_deceptive[word]+=1
                else:
                    positive_deceptive[word] = 1
            elif cls=='positive_truthful':
                if word in positive_truthful:
                    positive_truthful[word]+=1
                else:
                    positive_truthful[word] = 1
            elif cls=='negative_deceptive':
                if word in negative_deceptive:
                    negative_deceptive[word]+=1
                else:
                    negative_deceptive[word] = 1
            elif cls=='negative_truthful':
                if word in negative_truthful:
                    negative_truthful[word]+=1
                else:
                    negative_truthful[word] = 1
            
    return final_string, w_count


word_review_count =[0,0,0,0] #{'positive_deceptive': 0, 'positive_truthful': 0, 'negative_deceptive': 0, 'negative_truthful': 0}

for i in train_by_class['positive_polaritydeceptive_from_MTurk']:
    f = open(i,'r')
    for line in f:
        cleaned_string,sum_words = pre_processing(line,cls = 'positive_deceptive')
        word_review_count[0]+=sum_words
        #file.write(cleaned_string)
    f.close()
    
for i in train_by_class['positive_polaritytruthful_from_TripAdvisor']:
    f = open(i,'r')
    for line in f:
        cleaned_string,sum_words = pre_processing(line,cls = 'positive_truthful')
        word_review_count[1]+=sum_words
        #file.write(cleaned_string)
    f.close()

for i in train_by_class['negative_polaritydeceptive_from_MTurk']:
    f = open(i,'r')
    for line in f:
        cleaned_string,sum_words = pre_processing(line,cls = 'negative_deceptive')
        word_review_count[2]+=sum_words
        #file.write(cleaned_string)
    f.close()

for i in train_by_class['negative_polaritytruthful_from_Web']:
    f = open(i,'r')
    for line in f:
        cleaned_string,sum_words= pre_processing(line,cls = 'negative_truthful')
        word_review_count[3]+=sum_words
        #file.write(cleaned_string)
    f.close()

class_label_doc = [0,0,0,0]
def posterior_probablity():
    total_documents = 0
    for k in train_by_class.keys():
        if k == 'positive_polaritydeceptive_from_MTurk':
            class_label_doc[0] = len(train_by_class[k])
        elif k == 'positive_polaritytruthful_from_TripAdvisor':
            class_label_doc[1] = len(train_by_class[k])
        elif k == 'negative_polaritydeceptive_from_MTurk':
            class_label_doc[2] = len(train_by_class[k])
        elif k == 'negative_polaritytruthful_from_Web':
            class_label_doc[3] = len(train_by_class[k])
        total_documents+=len(train_by_class[k])
    for k in range(len(class_label_doc)):
        class_label_doc[k] = class_label_doc[k]/float(total_documents)
        
    return class_label_doc
posterior_probablity()

def classifier():
    fhandle = open('nbmodel.txt', 'w')
    l = len(vocab)
    model_dict = {}
    for k in vocab.keys():
        val = [0,0,0,0]
        for i in range(len(word_review_count)):
            if i==0:
                if k in positive_deceptive:
                    val[i] = np.log(class_label_doc[i]) + np.log((positive_deceptive[k]+1)/ (word_review_count[i]+l))    
                else:
                    val[i] = np.log(class_label_doc[i]) + np.log(1/ (word_review_count[i]+l))
            elif i==1:
                if k in positive_truthful:
                    val[i] = np.log(class_label_doc[i]) + np.log((positive_truthful[k]+1)/ (word_review_count[i]+l))
                else:
                    val[i] = np.log(class_label_doc[i]) + np.log(1/ (word_review_count[i]+l))
            elif i==2:
                if k in negative_deceptive:
                    val[i] = np.log(class_label_doc[i]) + np.log((negative_deceptive[k]+1)/ (word_review_count[i]+l))
                else:
                    val[i] = np.log(class_label_doc[i]) + np.log(1/ (word_review_count[i]+l))
            elif i==3:
                if k in negative_truthful:
                    val[i] = np.log(class_label_doc[i]) + np.log((negative_truthful[k]+1)/ (word_review_count[i]+l))
                else:
                    val[i] = np.log(class_label_doc[i]) + np.log(1/ (word_review_count[i]+l))
                
        model_dict[k] = [val[0], val[1], val[2], val[3]]
    fhandle.write(json.dumps(model_dict, indent=2))
    fhandle.close()
            
#Running my classifier
classifier()
