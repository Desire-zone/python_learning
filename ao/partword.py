# -*- coding: utf-8 -*-

import jieba
class Partword:
    def part_word(self,string):
        string = string.replace('\n','') 
        string = string.replace(' ','')
        string = string.replace('\r','')
        s = ''
        seg_list = jieba.cut(string)  # 默认是精确模式
        for j in seg_list:
            if j != '。'and j !='！':
              s = s +j + '\tO\n'
            else:
              s = s +j + '\tO\n\n'
        return s
    def save(self,string):
        path = './test/test.txt'
        file = open(path,'w',encoding = 'utf-8')
        file.write(string)
        file.close()
