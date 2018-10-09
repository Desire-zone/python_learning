# -*- coding: utf-8 -*-

import jieba
import time
time_start=time.time()
with open(
'E:/relabel.txt','r',encoding='utf-8') as flabel,open(
'E:/raw_segement_hotel_init.txt','r',encoding='utf-8') as fin1,open(
'E:/result0.txt','w') as fout:
    convertor=dict()
    labellist=[]
#    print(len(labellist))
#    print(labellist[:3])
    for label in flabel.readlines():
        cut=' '.join(jieba.cut(label.strip()))
        convertor[cut]=label.strip()
        labellist.append(cut)
        
    def recovery(raw_line):#标签恢复函数
        for label in labellist:
            if label in raw_line:
                raw_line=raw_line.replace(label,convertor[label])
        return raw_line
    
    all_lines=[]
    for line in fin1.readlines():
        cut_line=' '.join(jieba.cut(line.strip(),HMM=False))
        all_lines.append(recovery(cut_line))
    for line in all_lines:
        fout.write(line)
        fout.write('\n')
time_end=time.time()
print(time_end-time_start)