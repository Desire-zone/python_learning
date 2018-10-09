# -*- coding: utf-8 -*-
from instance import Instance
class Reader:
    def readfiles(self, path):
        tempstr = ''
        Insts = []
        try:    
            file = open(path, 'r',encoding='utf-8')
        except IOError:
            print ('文件读取异常')
        else:
            inst = Instance()
            for line in file.readlines():
                line = line.strip()
                if line == ''  :
                    Insts.append(inst)
                    inst = Instance()
                else:
                    for i in range(len(line)):
                        if line[i] != '\t' and i != len(line)-1:
                            tempstr += line[i]
                        else:
                            if line[i] == '\t':
                                inst.words.append(tempstr)     
                                tempstr = ''
                            else:
                                tempstr += line[i]
                                inst.labels.append(tempstr)
                                tempstr = ''
            file.close()
            if len(inst.words) != 0:
                Insts.append(inst)
            return Insts
