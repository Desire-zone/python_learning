# -*- coding: utf-8 -*-

from partword import Partword
import tkinter as tk
from tkinter import scrolledtext
from tkinter import filedialog
from tkinter import messagebox
from test import Test
class Window:    
    def __init__(self):
        self.var = ''
        self.windows = tk.Tk()
        self.windows.resizable()
       # width = tk.FALSE,height = tk.FALSE
        self.windows.title('意见要素识别')
        self.frame = tk.Frame(self.windows)
        self.iframe = tk.Frame(self.frame)
        self.oframe = tk.Frame(self.frame)
        self.var1 = tk.IntVar()
        self.var2 = tk.IntVar()
        self.itext = tk.scrolledtext.ScrolledText(self.iframe,width = 50,height = 40)
        self.otext = tk.scrolledtext.ScrolledText(self.oframe,width = 50,height = 40)
        self.otext.config(state = tk.DISABLED)
        self.label1 = tk.Label(self.frame)
        self.label2 = tk.Label(self.frame)
        self.label3 = tk.Label(self.frame)
        self.ilabel = tk.Label(self.frame,text = 'input:',width = 5,height = 2)
        self.olabel = tk.Label(self.frame,text = 'output:',width = 5,height = 2)
        self.button1 = tk.Button(self.frame,text='识别',command = self.getstr_and_show,width = 15,height = 2)
        self.button2 = tk.Button(self.frame,text='清空',command = self.clear,width = 15,height = 2)
        self.acheckbox = tk.Checkbutton(self.frame,text = '属性',variable = self.var1)
        self.echeckbox = tk.Checkbutton(self.frame,text = '评价',variable = self.var2)
        self.menubar = tk.Menu(self.windows)
        self.menubar.add_command(label = '打开',command = self.openfile)
        self.menubar.add_command(label = '保存',command = self.savefile)
        self.menubar.add_command(label = '另存为',command = self.save_asfile)
        self.menubar.add_command(label = '关闭',command = self.closefile)   
    def getstr_and_show(self):
        tempstr = ''
        strinput = self.itext.get('1.0',tk.END)
        if strinput == ''or strinput == '\r' or strinput == '\n':
           tk.messagebox.showinfo('提示','请输入要识别的句子')
           return
        self.otext.config(state = tk.NORMAL)
        self.otext.delete('1.0',tk.END)
        part = Partword()
        string = part.part_word(strinput)
        part.save(string)
        sentences = []
        strinput = strinput.replace(' ','')
        strinput = strinput.replace('\r','')
        strinput = strinput.replace('\n','')
        for i in strinput:
            if i != '。' and i != '！':
                tempstr += i
            else:
                tempstr += i
                if tempstr != '':
                    sentences.append(tempstr)
                tempstr = ''
        path_train = 'E:/20140517敖天宇/程序/corpus/raw_segement_hotel_final_train.txt'
        path_test = 'E:/20140517敖天宇/程序/test/test.txt'
        path_best = 'E:/20140517敖天宇/程序/best_model/best.pkl'
        tempstr = ''
        test = Test()
        test.prediction(path_test,path_train,path_best)
        if self.var1.get() == 0 and self.var2.get() == 0:
            tk.messagebox.showinfo('提示','请选择需要识别的属性或评价')
            return
        elif self.var1.get() == 0 and self.var2.get() == 1:
            if len(test.Eval) !=len(sentences):
                tk.messagebox.showinfo('提示','输入的句子请以句号或叹号结尾')
                return
            for i in range(len(test.Eval)):
                tempstr = tempstr + '第'+str(i+1)+'句:'+sentences[i]+'\n评价:'
                for j in range(len(test.Eval[i])):
                    tempstr = tempstr + test.Eval[i][j] + ' '
                tempstr += '\n'
        elif self.var1.get() == 1 and self.var2.get() == 0:
            if len(test.Attr) !=len(sentences):
                tk.messagebox.showinfo('提示','输入的句子请以句号或叹号结尾')
                return
            for i in range(len(test.Attr)):
                tempstr = tempstr + '第'+str(i+1)+'句:'+sentences[i]+'\n'+'属性:'
                for j in range(len(test.Attr[i])):
                    tempstr = tempstr + test.Attr[i][j]+' '
                tempstr += '\n'
        elif self.var1.get() == 1 and self.var2.get() == 1:
            if len(test.Eval) !=len(sentences):
                tk.messagebox.showinfo('提示','输入的句子请以句号或叹号结尾')
                return
            for i in range(len(test.Attr)):
                tempstr = tempstr + '第'+str(i+1)+'句:'+sentences[i]+'\n'+'属性:'
                for j in range(len(test.Attr[i])):
                    tempstr = tempstr + test.Attr[i][j]+' '
                tempstr = tempstr + '\n评价:'
                for j in range(len(test.Eval[i])):
                    tempstr = tempstr + test.Eval[i][j] + ' '
                tempstr += '\n'
        self.otext.insert('1.0',tempstr)
        self.otext.config(state = tk.DISABLED)
    def clear(self):
        self.acheckbox.deselect()
        self.echeckbox.deselect()
        self.itext.delete(1.0,tk.END)
        self.otext.config(state = tk.NORMAL)
        self.otext.delete(1.0,tk.END)
        self.otext.config(state = tk.DISABLED)
    def openfile(self):
        tempstr = ''
        filetypes = [  
                ("All Files", '*'),  
                ("Python Files", '*.py', 'TEXT'),  
                ("Text Files", '*.txt', 'TEXT'),  
                ("Config Files", '*.conf', 'TEXT')]
        try:
            fobj =  tk.filedialog.askopenfile(mode = 'rb',filetypes=filetypes)
        except Exception:
            print ('打开文件错误')
        else:
            if fobj != None:
                for line in fobj.readlines():
                    line = line.decode('utf-8')
                    tempstr +=line
                self.itext.delete('1.0', tk.END)  
                self.itext.insert('1.0', tempstr)  
    def savefile(self):
        path = self.var
        if path != '':  
            file = open(path, 'w',encoding = 'gbk')
            self.otext.config(state = tk.NORMAL)
            file.write(self.otext.get('1.0', tk.END))
            self.otext.config(state = tk.DISABLED)
            file.close()  
        else: 
            self.save_asfile()
    def save_asfile(self):
        filetypes = [  
               ("All Files", '*'),  
               ("Python Files", '*.py', 'TEXT'),  
               ("Text Files", '*.txt', 'TEXT'),  
               ("Config Files", '*.conf', 'TEXT')]
        self.otext.config(state = tk.NORMAL)
        text_value = self.otext.get('1.0', tk.END)
        self.otext.config(state = tk.DISABLED)
        if text_value != '':
            try:
                fobj = tk.filedialog.asksaveasfile(mode = 'w',filetypes=filetypes)  
            except IOError:
                print ('文件写入异常')
            else:
                self.var = fobj.name
                fobj.write(text_value) 
    def closefile(self):
        close = tk.messagebox.askokcancel('提示','真的要关闭吗')
        if  close == True:
            self.windows.destroy()
    def show(self):
        self.windows.config(menu = self.menubar)
        self.frame.grid(row = 0,column = 0,rowspan = 5,columnspan = 5)
        self.iframe.grid(row = 1,column = 0,rowspan = 2,columnspan = 1)
        self.oframe.grid(row = 1,column = 4,rowspan = 2,columnspan = 1)
        self.ilabel.grid(row = 0,column = 0,columnspan = 1)
        self.olabel.grid(row = 0,column = 4,columnspan = 1)
        self.itext.grid(row = 1,column = 0,rowspan = 2,columnspan = 1)
        self.otext.grid(row = 1,column = 4,rowspan = 2,columnspan = 1)
        self.acheckbox.grid(row = 1,column = 2)
        self.echeckbox.grid(row = 2,column = 2)
        self.label3.grid(row = 0,column = 3,rowspan = 5)
        self.label1.grid(row = 4,column = 0,columnspan = 5)
        self.button1.grid(row = 5,column = 0,columnspan = 1)
        self.button2.grid(row = 5,column = 4,columnspan = 1)
        self.label2.grid(row = 6,column = 0,columnspan = 5)
        self.windows.mainloop()