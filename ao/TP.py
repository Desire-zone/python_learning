# -*- coding: utf-8 -*-

from read import Reader
class textProcess:
    def __init__(self):
        self.sentence = []
        self.attr = []
        self.eval = []
        self.other = []
        self.word = []
        self.Allword = []
        self.Alabel = []
        self.Elabel = []
        self.Olabel = []
    def read(self,path):
        tempstr = ''
        try:    
            file = open(path,'r',encoding = 'utf-8')
        except IOError:
            print ('文件读取错误')
        else:
            for line in file.readlines():
                for w in line:
                    if w != ' ' and w != '\r' and w!='\n':
                        tempstr += w
                    else:
                        self.word.append(tempstr)
                        tempstr = ''
                for i in range(len(self.word)):
                    if self.word[i][0:2] == '<a':
                        self.word[i] = '<a'
                    if self.word[i][0:3] == '</a':
                        self.word[i] = '</a'
                    if self.word[i][0:2] == '<e':
                        self.word[i] = '<e'
                    if self.word[i][0:3] == '</e':
                        self.word[i] = '</e'
                self.Allword.append(self.word[0:len(self.word)])
                self.word.clear()
            file.close()
    def process(self):
        tempstr = ''
        for line in self.Allword:
            abegin = -1
            aend = -1
            ebegin = -1
            eend = -1 
            for i in range(len(line)):                
                    if  line[i][0:2] == '<a':
                        abegin = i+1
                    if line[i][0:3] == '</a':
                        aend = i         
                        if abegin >= 0 :
                            for j in range(abegin,aend):
                                self.attr.append(line[j])
                            if len(self.attr) == 1:
                                self.Alabel.append('S-A')
                            elif len(self.attr) == 2:
                                self.Alabel.append('B-A')
                                self.Alabel.append('E-A')
                            else:
                                self.Alabel.append('B-A')
                                for j in range(1,len(self.attr)-1):
                                    self.Alabel.append('M-A')
                                self.Alabel.append('E-A')
                            for k in range(len(self.attr)):
                                tempstr = tempstr + self.attr[k] +'\t' + self.Alabel[k] + '\n'
                            self.attr.clear()
                            self.Alabel.clear()
                            abegin = -1
                            aend = -1
                            
                    if line[i][0:2] == '<e' :
                        ebegin = i+1
                    if line[i][0:3] == '</e':
                        eend = i
                        if ebegin >= 0:
                            for j in range(ebegin,eend):
                                self.eval.append(line[j])
                            if len(self.eval) == 1:
                                self.Elabel.append('S-E')
                            elif len(self.eval) == 2:
                                self.Elabel.append('B-E')
                                self.Elabel.append('E-E')
                            else:
                                self.Elabel.append('B-E')
                                for j in range(1,len(self.eval)-1):
                                    self.Elabel.append('M-E')
                                self.Elabel.append('E-E')
                            for k in range(len(self.eval)):
                                tempstr = tempstr + self.eval[k] +'\t' + self.Elabel[k] + '\n'
                            self.eval.clear()
                            self.Elabel.clear()
                            ebegin = -1
                            eend = -1          
                    if (abegin <0 and ebegin < 0 ) \
                        and line[i][0:10] != '</a' and line[i][0:10] != '</e':
                        tempstr = tempstr + line[i][0:10] +'\t' + 'O' + '\n'
           
            tempstr += '\n'
           
        return tempstr
    def writefile(self,string,path):
        try:    
            file = open(path,'w',encoding = 'utf-8')
        except IOError:
            print ('文件写入错误')
        else:
            file.write(string)
            file.close()
ipath = 'E:/raw_segement_hotel_part.txt'
opath = 'E:/raw_segement_hotel_result_final.txt'
tp = textProcess()
tp.read(ipath)
temp = tp.process()
tp.writefile(temp,opath)
