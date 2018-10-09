# -*- coding: utf-8 -*-
from hyperparameter import Hyperparameter
from example import Example
from read import Reader
from alphabet import AlphaBet
import torch
import torch.nn.functional as F
import random
from model import BiLstm
from evaluate import Eval
class NER:
    def __init__(self):
        self.word_AlphaBet = AlphaBet()
        self.label_AlphaBet = AlphaBet()
        self.hyperpara = Hyperparameter()
        
    def create_alphabet(self, file_train):
        word_list = []
        label_list = []
        for inst in file_train:
            for m in inst.words:
                if m not in word_list:
                    word_list.append(m)
            for n in inst.labels:
                if n not in label_list:
                    label_list.append(n)
        word_list.append(self.hyperpara.unknow)
        word_list.append(self.hyperpara.padding)
        label_list.append(self.hyperpara.padding)
        self.word_AlphaBet.makeVocab(word_list)  
        self.label_AlphaBet.makeVocab(label_list)  
        self.hyperpara.unknow_id = self.word_AlphaBet.dict[self.hyperpara.unknow]
        self.hyperpara.padding_id = self.word_AlphaBet.dict[self.hyperpara.padding]
        self.hyperpara.tag_padding_id = self.label_AlphaBet.dict[self.hyperpara.padding]
        self.hyperpara.embedding_num = len(word_list)
        self.hyperpara.tag_size = len(label_list)
        
    def change(self, insts):
        exams = []
        for inst in insts:
            example = Example()
            for w in inst.words:
                if w in self.word_AlphaBet.list:
                    example.wordIndexs.append(self.word_AlphaBet.dict[w])
                else:
                    example.wordIndexs.append(self.hyperpara.unknow_id)      
            for l in inst.labels:
                labelId = self.label_AlphaBet.dict[l]
                example.labelIndexs.append(labelId)
            exams.append(example)
        return exams    #每句话的句子和标签的ID
  
    def variable(self, batch_list):
        maxlength = 0
        for i in range(len(batch_list)):
            if maxlength < len(batch_list[i].wordIndexs):
                maxlength = len(batch_list[i].wordIndexs)
        x = torch.LongTensor(len(batch_list), maxlength)
        x = torch.autograd.Variable(x)
        y = torch.LongTensor(len(batch_list) * maxlength)
        y = torch.autograd.Variable(y)
        for i in range(len(batch_list)):
            for n in range(len(batch_list[i].wordIndexs)):
                x.data[i][n] = batch_list[i].wordIndexs[n]
                y.data[(i * maxlength) + n] = batch_list[i].labelIndexs[n]
            for n in range(len(batch_list[i].wordIndexs), maxlength):
                x.data[i][n] = self.hyperpara.padding_id
                y.data[(i * maxlength) + n] = self.hyperpara.tag_padding_id
        return x, y

    def train(self, path_train, path_dev, path_test,path_PRF,path_model,path_bestModel):
        #读取训练集、测试集、开发集 并 建立字典
        reader = Reader()
        traininsts = reader.readfiles(path_train)
        devinsts = reader.readfiles(path_dev)
        testinsts = reader.readfiles(path_test)
        print('Training Instance:', len(traininsts))
        print('Dev Instance:', len(devinsts))
        print('Test Instance:', len(testinsts))
        self.create_alphabet(traininsts)
        
        #字符串转成ID
        trainExamples = self.change(traininsts)  # e_train
        devExamples = self.change(devinsts)
        testExamples = self.change(testinsts)
        
        self.model = BiLstm(self.hyperpara)  # BiLstm模型
        # 加载模型
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
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hyperpara.lr)  # 优化器
        total_num = len(trainExamples)
        for epoch in range(1, self.hyperpara.epochs):
            print("————————第{}轮迭代，共{}轮————————".format(epoch, self.hyperpara.epochs))
            total = 0
            random.shuffle(trainExamples)  # 随机打乱训练集顺序，能有效提高准确率
            try:    
                part = total_num // self.hyperpara.batch
                if total_num % self.hyperpara.batch != 0:
                    part += 1
            except ZeroDivisionError:
                print ('batch数为0，除0错误')
            else:
                
            #开始训练
                self.model.train()
                for idx in range(part):
                    begin = idx * self.hyperpara.batch
                    end = (idx + 1) * self.hyperpara.batch
                    if end > total_num:
                        end = total_num
                    batch_list = []
       #             batch_list_len = []
                    for idy in range(begin, end):
                        batch_list.append(trainExamples[idy])
      #                  batch_list_len.append(len(trainExamples[idy].wordIndexs))
                    optimizer.zero_grad()
                    x, y = self.variable(batch_list)           
                    lstm_feats = self.model(x)  
                    loss = F.cross_entropy(lstm_feats, y)
                    total += 1
                    loss.backward()
                    optimizer.step()
                    print('current:', total, ", loss:", loss.data[0])
            #开发集测试效果
            eval_dev = Eval()
            eval_dev_A = Eval()
            eval_dev_E = Eval()
            for idx in range(len(devExamples)):
                dev_list = []
                dev_list.append(devExamples[idx])
                x, y = self.variable(dev_list)
                lstm_feats = self.model(x)
                predict = self.getMaxIndex(lstm_feats)  
                predictLabels = []
                for idy in range(len(predict)):
                    predictLabels.append(self.label_AlphaBet.list[predict[idy]])
                gold_ent,predict_ent = devinsts[idx].evalPRF(predictLabels, eval_dev)
                gold_ent_A,gold_ent_E,predict_ent_A,predict_ent_E = devinsts[idx].getAE(gold_ent,predict_ent)
                devinsts[idx].evalAEPRF(gold_ent_A,predict_ent_A,eval_dev_A)
                devinsts[idx].evalAEPRF(gold_ent_E,predict_ent_E,eval_dev_E)
            line = ''
            print('Dev: ', end="")
            d_precision,d_recall,d_fscore =eval_dev.getFscore()
            line = line+str(epoch) +'.dev:\nP:'+('%.2f' % (d_precision*100))+' R:'+('%.2f' % (d_recall*100))+' F:'+('%.2f' % (d_fscore*100))+'\n'
            print("precision:" , d_precision*100, ", recall: " ,d_recall*100, ", fscore:"  , d_fscore*100)
            d_precision,d_recall,d_fscore =eval_dev_A.getFscore()
            line = line +'A_P:'+('%.2f' % (d_precision*100))+' A_R:'+('%.2f' % (d_recall*100))+' A_F:'+('%.2f' %( d_fscore*100))+'\n'
            print("precision:" , d_precision*100, ", recall: " ,d_recall*100, ", fscore:"  , d_fscore*100)
            d_precision,d_recall,d_fscore =eval_dev_E.getFscore()
            line = line +'E_P:'+('%.2f' % (d_precision*100))+' E_R:'+('%.2f' % (d_recall*100))+' E_F:'+('%.2f' % (d_fscore*100))+'\n'
            print("precision:" , d_precision*100, ", recall: " ,d_recall*100, ", fscore:"  , d_fscore*100)
            #测试集测试效果
            eval_test = Eval()
            eval_test_A = Eval()
            eval_test_E = Eval()
            for idx in range(len(testExamples)):
                test_list = []
                test_list.append(testExamples[idx])
                x, y = self.variable(test_list)                
                lstm_feats = self.model(x)
                predict = self.getMaxIndex(lstm_feats)
                predictLabels = []
                for idy in range(len(predict)):
                    predictLabels.append(self.label_AlphaBet.list[predict[idy]])
                gold_ent,predict_ent = testinsts[idx].evalPRF(predictLabels, eval_test)
                gold_ent_A,gold_ent_E,predict_ent_A,predict_ent_E = testinsts[idx].getAE(gold_ent,predict_ent)
                testinsts[idx].evalAEPRF(gold_ent_A,predict_ent_A,eval_test_A)
                testinsts[idx].evalAEPRF(gold_ent_E,predict_ent_E,eval_test_E)                 
            print('Test: ', end="")
            t_precision,t_recall,t_fscore =eval_test.getFscore()
            line = line +'test:\nP:'+('%.2f' % (t_precision*100))+' R:'+('%.2f' % (t_recall*100))+' F:'+('%.2f' % (t_fscore*100))+'\n'
            print("precision:" , t_precision*100, ", recall: " ,t_recall*100, ", fscore:"  , t_fscore*100)
            t_precision,t_recall,t_fscore =eval_test_A.getFscore()
            line = line +'A_P:'+('%.2f' % (t_precision*100))+' A_R:'+('%.2f' % (t_recall*100))+' A_F:'+('%.2f' % (t_fscore*100))+'\n'
            print("precision:" , t_precision*100, ", recall: " ,t_recall*100, ", fscore:"  , t_fscore*100)
            t_precision,t_recall,t_fscore =eval_test_E.getFscore()
            line = line +'E_P:'+('%.2f' % (t_precision*100))+' E_R:'+('%.2f' %( t_recall*100))+' E_F:'+('%.2f' % (t_fscore*100))+'\n'
            print("precision:" , t_precision*100, ", recall: " ,t_recall*100, ", fscore:"  , t_fscore*100) 
            #保存模型
            if self.hyperpara.save_pattern == 0:
                torch.save(self.model.state_dict(),path_model+str(epoch)+'.pkl')
            elif self.hyperpara.save_pattern == 1:
                torch.save(self.model,path_model+str(epoch)+'.pkl')
            try:    
                file = open(path_PRF,'a+',encoding = 'utf-8')
            except IOError:
                print ('文件读取异常')
            else:    
                file.write(line)
                file.close()
    def getMaxIndex(self, score):
        predict = []
        sentence_len = score.size()[0]
        labelsize = score.size()[1]       
        for iid in range(sentence_len):
            mmax = score.data[iid][0]
            maxindex = 0
            for idx in range(labelsize):
                tmp = score.data[iid][idx]
                if tmp > mmax:
                    mmax = tmp
                    maxindex = idx
            predict.append(maxindex)
        return predict

#path_train = 'E:/20140517敖天宇/程序/corpus/raw_segement_hotel_final_train.txt'
#path_dev = 'E:/20140517敖天宇/程序/corpus/raw_segement_hotel_final_dev.txt'
#path_test = 'E:/20140517敖天宇/程序/corpus/raw_segement_hotel_final_test.txt'
#path_PRF = 'E:/20140517敖天宇/程序/now_model/PRF.txt'
#path_model = 'E:/20140517敖天宇/程序/now_model/'
#path_bestModel = 'E:/20140517敖天宇/程序/best_model/best.pkl'
#a = NER()
#a.train(path_train, path_dev, path_test,path_PRF,path_model,path_bestModel)
