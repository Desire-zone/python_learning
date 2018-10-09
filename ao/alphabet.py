# -*- coding: utf-8 -*-
from collections import OrderedDict
class AlphaBet:
    def __init__(self):
        self.list = []
        self.dict = OrderedDict()

    def makeVocab(self, inst):
        for i in inst:
            self.list.append(i)
        for k in range(len(inst)):
            self.dict[inst[k]] = k
			