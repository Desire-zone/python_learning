# -*- coding: utf-8 -*-

import chardet as cd
from read import Reader
class Tool:
    def __init__(self):
        pass
	
#    def codetransition(self,path):
#        file = open(path,'rb')
#        data = file.read()
#        file.close()
#        eencode = cd.detect(data)
#        print (eencode)
#        if eencode['encoding'] == None:
#            data = data.decode('utf-8').encode('utf-8')
#            data = data.decode('utf-8')
#        elif eencode['encoding'] == 'ascii':
#            data = data.decode('ascii').encode('utf-8')
#            data = data.decode('utf-8')
#        elif eencode['encoding'] == 'UTF-8-SIG':
#            data = data.decode('UTF-8-SIG').encode('utf-8')
#            data = data.decode('utf-8')
#        elif eencode['encoding'] == 'GB2312':
#            data = data.decode('GB2312').encode('utf-8')
#            data = data.decode('utf-8')
#        elif eencode['encoding'] == 'UTF-8-SIG':
#            data = data.decode('ISO-8859-1').encode('utf-8')
#            data = data.decode('utf-8')
#        print (data)
#        file = open(path,'w',encoding = 'utf-8')
#        file.write(data)
#        file.close()
	
    def findmaxFscore(self,path):
        count = 1
        tmp = ''
        F_score = []
        try:    
            file = open(path,'r',encoding = 'utf-8')
        except IOError:
            print ('文件读取错误')
        else:    
            for line in file.readlines():
                if (count - 2) % 8 == 0 :
                    for i in range(len(line)):
                        if line[i] == 'F':
                            for j in range(i+2,len(line)):                 
                                tmp += line[j]     
                            F_score.append(float(tmp))
                            tmp = ''
                            break
                count += 1
        count = 2
        mmax = F_score[0]
        for i in range(1,len(F_score)):
            if F_score[i] > mmax:
                mmax = F_score[i]
                count =2+8* i
        return mmax,count
    def getAEOcount(self,path):
        a = 0
        e = 0
        o = 0
        reader = Reader()
        Inst = reader.readfiles(path)
        for i in Inst:
            for j in range(len(i.labels)):   
               if len(i.labels[j]) == 3:
                    if i.labels[j][2] == 'A':
                        a = a+1
                    elif i.labels[j][2] == 'E':
                        e = e+1
               elif i.labels[j] == 'O':
                    o += 1
        return a,e,o
path = 'E:/20140517敖天宇/程序/now_model/PRF.txt'
tool = Tool()
tool1 = Tool()
mmax,count=tool.findmaxFscore(path)
print (mmax,count)
#a,e,o=tool1.getAEOcount(path)
#print (a,e,o)


