# -*- coding: utf-8 -*-
class Instance:
    def __init__(self):
        self.words = []
        self.labels = []
        self.predict_entt = []
      
    def evalPRF(self, predict_labels, evaluate):
        gold_ent = self.get_ent(self.labels)
        predict_ent = self.get_ent(predict_labels)       
        self.predict_entt = predict_ent
        evaluate.predict_num += len(predict_ent)
        evaluate.gold_num += len(gold_ent)
        for p in predict_ent:
            if p in gold_ent:
                evaluate.correct_num += 1
        return gold_ent,predict_ent
    
    def getAE(self,gold_ent,predict_ent):
        gold_ent_A = []
        gold_ent_E = []
        predict_ent_A = []
        predict_ent_E = []
        for p in gold_ent:
            if p in gold_ent and p[0] == 'A':
                gold_ent_A.append(p)
            elif p in gold_ent and p[0] == 'E':
                gold_ent_E.append(p)
        for p in predict_ent:
            if p in predict_ent and p[0] == 'A':
                predict_ent_A.append(p)
            elif p in predict_ent and p[0] == 'E':
                predict_ent_E.append(p)
        return gold_ent_A,gold_ent_E,predict_ent_A,predict_ent_E
    
    def evalAEPRF(self,gold_ent,predict_ent,evaluate):
        evaluate.predict_num += len(predict_ent)
        evaluate.gold_num += len(gold_ent)
        for p in predict_ent:
            if p in gold_ent:
                evaluate.correct_num += 1    
        
    def get_ent(self, labels):
        idx = 0
        ent = []
        while(idx < len(labels)):
            if (self.is_start_label(labels[idx])):
                idy = idx
                endpos = -1
                while(idy < len(labels)):
                    if not self.is_continue_label(labels[idy], labels[idx], idy - idx):
                        endpos = idy - 1
                        break
                    endpos = idy
                    idy += 1
                ent.append(self.cleanLabel(labels[idx]) + '[' + str(idx) + ',' + str(endpos) + ']')
                idx = endpos
            idx += 1
        return ent

    def cleanLabel(self, label):
        start = ['B', 'b', 'M', 'm', 'E', 'e', 'S', 's', 'O', 'o']
        if len(label) > 2 and label[1] == '-':
            if label[0] in start:
                return label[2:]
        return label

    def is_continue_label(self, label, startLabel, distance):
        if distance == 0:
            return True
        if len(label) < 3:
            return False
        if distance != 0 and self.is_start_label(label):
            return False
        if (startLabel[0] == 's' or startLabel[0] == 'S') and startLabel[1] == '-':
            return False
        if self.cleanLabel(label) != self.cleanLabel(startLabel):
            return False
        return True

    def is_start_label(self, label):
        start = ['b', 'B', 's', 'S']
        if(len(label) < 3):
            return False
        else:
            return (label[0] in start) and label[1] == '-'
    def extractA_and_E(self):
        Attr = []
        Eval = []
        word = ''
        digit = ''
        start = 0
        end = 0
        for i in range(len(self.predict_entt)):
            count = 0  
            if self.predict_entt[i][0] == 'A' or self.predict_entt[i][0] == 'a':
                for j in range (2,len(self.predict_entt[i])):
                    if self.predict_entt[i][j] >= '0' and self.predict_entt[i][j] <= '9':
                        digit += self.predict_entt[i][j]
                    else:
                        if count >= 1:
                            end = int(digit)
                            digit = ''
                        else:
                            count += 1
                            start = int(digit)
                            digit = ''
                
                for k in range (start,end+1):
                    word += self.words[k]
                Attr.append(word)
                word = ''
            elif self.predict_entt[i][0] == 'E' or self.predict_entt[i][0] == 'e':
                for j in range (2,len(self.predict_entt[i])):
                    if self.predict_entt[i][j] >= '0' and self.predict_entt[i][j] <= '9':
                        digit += self.predict_entt[i][j]
                    else:
                        if count >= 1:
                            end = int(digit)
                            digit = ''
                        else:
                            count += 1
                            start = int(digit)
                            digit = ''
                for k in range (start,end+1):
                    word += self.words[k]
                Eval.append(word)
                word = ''                 
        return Attr,Eval
