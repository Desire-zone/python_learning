# -*- coding: utf-8 -*-
class Hyperparameter:
    def __init__(self):
        self.lr = 1e-3
        self.batch = 32
        self.epochs = 64
        self.embedding_dim = 200
        self.LSTM_hidden_dim = 400
        self.num_layers = 2
        self.inputdropout = 0
        self.outputdropout = 0.4
        self.shuffle = 'True'
        self.loadModel = 1
        self.save_pattern = 0
        self.load_pattern = 1
        self.embedding_num = 0
        self.tag_size = 0
        self.unknow = "#-unknow-#"
        self.unknow_id = 0
        self.padding = "#-padding-#"
        self.tag_padding_id = 0
        self.padding_id = 0
