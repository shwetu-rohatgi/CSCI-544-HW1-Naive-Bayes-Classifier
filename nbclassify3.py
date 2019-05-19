# use this file to classify using naive-bayes classifier 
# Expected: generate nboutput.txt

import sys
import glob
import os
import collections
import re
import numpy as np
import json

all_files = glob.glob(os.path.join(sys.argv[1], '*/*/*/*.txt'))

model_dict = open("nbmodel.txt", 'r').read()
model_dict = json.loads(model_dict)
        
def preprocessing(content_data):
    content_data = content_data.replace('\n','')
    content_data = content_data.replace('\t','')
    #content_data = content_data.lower()
    
    cleaned_line = re.sub('[^a-z\s]+',' ',content_data)
    
    cleaned_line = re.sub('(\s+)',' ',cleaned_line)
    
    stop_words = ['1','2','3','4','5','6','7','8','9','0','it', 'hers', 'between', 'yourself', 'but', 'the','again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are','his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'up', 'to', 'ours', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than','hotel','stay','i','we', 'these', 'your','while', 'above', 'both','where', 'too', 'only','had', 'she', 'all','do', 'its', 'yours', 'such','chicago','day','ourselves','no', 'when', 'at', 'any','who', 'as', 'from']

    final_string = ""
    for word in cleaned_line.split():        
        if word in stop_words:
            continue
        else:
            final_string = final_string+" "+word
    return final_string

output = open('nboutput.txt', 'w')
for file in all_files:
    p = [0,0,0,0]
    f = open(file, 'r')
    data = preprocessing(f.read()).split()
    for eachword in data:
        if eachword in model_dict:
            p[0] += model_dict[eachword][0]
            p[1] += model_dict[eachword][1]
            p[2] += model_dict[eachword][2]
            p[3] += model_dict[eachword][3]
    max_label = max(p)
    label_max_value = p.index(max_label)
    if label_max_value==0:
        output.write("deceptive positive "+str(file)+"\n")
    elif label_max_value==1:
        output.write("truthful positive "+str(file)+"\n")
    elif label_max_value==2:
        output.write("deceptive negative "+str(file)+"\n")
    elif label_max_value==3:
        output.write("truthful negative "+str(file)+"\n")
                
output.close()