# -*- coding: utf-8 -*-
from hyperparameter import Hyperparameter
from read import Reader
import torch
from model import BiLstm
from evaluate import Eval
from train import NER
class Test:
    def __init__(self):
        self.Attr = []
        self.Eval = []
    def prediction(self,path_test,path_train,path_bestModel):
        self.hyperpara = Hyperparameter()
        eval_test = Eval()
        ner = NER()
        reader = Reader()
        traininsts = reader.readfiles(path_train)
        testinsts = reader.readfiles(path_test)
        ner.create_alphabet(traininsts)
        self.hyperpara.tag_size = ner.hyperpara.tag_size
        self.hyperpara.embedding_num = ner.hyperpara.embedding_num
        self.model = BiLstm(self.hyperpara)  # BiLstm模型
        if self.hyperpara.loadModel == 1 and\
           self.hyperpara.load_pattern ==  1:
               try:       
                   self.model.load_state_dict(torch.load(path_bestModel))
               except Exception:
                  print ('模型参数不匹配')
               else:
                   pass
        elif self.hyperpara.loadModel == 1 and\
             self.hyperpara.load_pattern == 0 :
                 try:    
                     self.model = torch.load(path_bestModel)
                 except Exception:
                    print ('模型参数不匹配')
                 else:
                     pass  
        testExamples = ner.change(testinsts)
        for idx in range(len(testExamples)):
            test_list = []
            test_list.append(testExamples[idx])
            x, y = ner.variable(test_list)    
            lstm_feats = self.model(x)        
            predict = ner.getMaxIndex(lstm_feats)
            predictLabels = []
            for idy in range(len(predict)):
                predictLabels.append(ner.label_AlphaBet.list[predict[idy]])
            testinsts[idx].evalPRF(predictLabels, eval_test)
            a,e = testinsts[idx].extractA_and_E()
            self.Attr.append(a)
            self.Eval.append(e)
			