import torch
import numpy
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable
import matplotlib.image as mpimg
import tkinter as tk
import tkinter.messagebox
#import matplotlib.pyplot as plt

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=16,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(         # input shape (1, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output, x    # return x for visualization
#load parameters
net = CNN()
try:
    net.load_state_dict(torch.load('net_params.pkl'))
except Exception as e:
    print("fail to load parameters!")
    exit(0)
#create window
root = tk.Tk()
root.title('CNN Test')
root.geometry('460x405')
root.geometry("+500+200")
#add bg color
frame=tk.Frame(
    height=700,
    width=1000,
    bg='#C0C0C0',
).pack(expand='YES',fill='both')
#please input image name!
l0 = tk.Label(root,
    text='请输入图片名称：',
    font=('Arial',15),
    bg='#C0C0C0',
    fg='black'
    ).place(x=10,y=120)
#input entry
e = tk.Entry(root,
             bg='#909090',
             )
e.place(x=10,y=160)
#view the result
predict = tk.StringVar()
path = './test_image/'
def pre():
    var = e.get()
    try:
        img = Image.open(path+var).convert('L')
    except Exception as ex:
        showwrong()
        clear()
        return
    img = img.resize((28, 28))
    data = numpy.array(img).reshape(1, 1, 28, 28) / 255.0
    torch_data = torch.from_numpy(data)
    test_x = Variable(torch_data).type(torch.FloatTensor)
    test_output, _ = net(test_x)
    numpy_out_data=test_output.data.numpy().squeeze()
    #softmax
    exp_out=numpy.exp(numpy_out_data)
    softmax_out=exp_out/exp_out.sum()
    for i in range(10):
        softmax_out[i]=round(softmax_out[i],4)
        softmax_out[i]=softmax_out[i]*100
        softmax_out_dot[i].set('%%%s'%(softmax_out[i]))
        #print(softmax_out_dot[i])
        #print(softmax_out[i])
    pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
    predict.set(pred_y)
#connect button '清空'
def clear():
    e.delete(0,30)
#connect button '显示'
def open_image():
    var = e.get()
    try:
        img_open = Image.open(path+var)
    except Exception as ex:
        showwrong()
        clear()
        return
    img_open.show()
#弹窗
def showwrong():
   tk.messagebox.showinfo(title='Attention!',
                          message='input error,please input again',
                          )
#predict lable
l = tk.Label(root,
             textvariable=predict,
             bg='green',
             font=('Arial',47),
             width=3,
             height=3).place(x=220,y=45)
#probability of 0~9
ll = tk.Label(root,
              text="Probability of 0 to 9:",
              font=('Arial',35),
              bg='#C0C0C0',
              ).place(x=20,y=320)
#predict
lll = tk.Label(root,
              text="predict:",
              font=('Arial',22),
              bg='#C0C0C0',
              ).place(x=222,y=6)

#识别button
b1 = tk.Button(root,
               text="识别",
               command=pre,
               width=6,
               bg='#FFFF00',
               height=2).place(x=230,y=240)
#清空button
b2 = tk.Button(root,
               text="清空",
               command=clear,
               width=6,
               height=2).place(x=121,y=240)
#显示图片button
b3 = tk.Button(root,
               text="显示",
               command=open_image,
               width=6,
               height=2).place(x=10,y=240)
softmax_out_dot = []
number = []
for i in range(10):
    softmax_out_dot.append(tk.StringVar())
    number.append(tk.StringVar())
k=0
#show predict probability
l0=tk.Label(root,
         textvariable=softmax_out_dot[0],
         bg='green',
         fg='red',
         font=('Arial',15),
         width=5,
         height=1).place(x=400,y=5)
l0_number=tk.Label(root,
         textvariable=number[0],
         font = ('Arial', 15),
         width = 5,
         fg='black',
         bg='#C0C0C0',
         height= 1 ).place(x=350,y=5)
number[0].set(0)

l1=tk.Label(root,
         textvariable=softmax_out_dot[1],
         bg='green',
         fg='red',
         font=('Arial',15),
         width=5,
         height=1).place(x=400,y=45)
l1_number=tk.Label(root,
         textvariable=number[1],
         font = ('Arial', 15),
         width = 5,
         fg='black',
         bg='#C0C0C0',
         height= 1 ).place(x=350,y=45)
number[1].set(1)

l2=tk.Label(root,
         textvariable=softmax_out_dot[2],
         bg='green',
         fg='red',
         font=('Arial',15),
         width=5,
         height=1).place(x=400,y=85)
l2_number=tk.Label(root,
         textvariable=number[2],
         font = ('Arial', 15),
         width = 5,
         fg='black',
         bg='#C0C0C0',
         height= 1 ).place(x=350,y=85)
number[2].set(2)

l3=tk.Label(root,
         textvariable=softmax_out_dot[3],
         bg='green',
         fg='red',
         font=('Arial',15),
         width=5,
         height=1).place(x=400,y=125)
l3_number=tk.Label(root,
         textvariable=number[3],
         font = ('Arial', 15),
         width = 5,
         fg='black',
         bg='#C0C0C0',
         height= 1 ).place(x=350,y=125)
number[3].set(3)

l4=tk.Label(root,
         textvariable=softmax_out_dot[4],
         bg='green',
         fg='red',
         font=('Arial',15),
         width=5,
         height=1).place(x=400,y=165)
l4_number=tk.Label(root,
         textvariable=number[4],
         font = ('Arial', 15),
         width = 5,
         fg='black',
         bg='#C0C0C0',
         height= 1 ).place(x=350,y=165)
number[4].set(4)

l5=tk.Label(root,
         textvariable=softmax_out_dot[5],
         bg='green',
         fg='red',
         font=('Arial',15),
         width=5,
         height=1).place(x=400,y=205)
l5_number=tk.Label(root,
         textvariable=number[5],
         font = ('Arial', 15),
         width = 5,
         fg='black',
         bg='#C0C0C0',
         height= 1 ).place(x=350,y=205)
number[5].set(5)

l6=tk.Label(root,
         textvariable=softmax_out_dot[6],
         bg='green',
         fg='red',
         font=('Arial',15),
         width=5,
         height=1).place(x=400,y=245)
l6_number=tk.Label(root,
         textvariable=number[6],
         font = ('Arial', 15),
         width = 5,
         fg='black',
         bg='#C0C0C0',
         height= 1 ).place(x=350,y=245)
number[6].set(6)

l7=tk.Label(root,
         textvariable=softmax_out_dot[7],
         bg='green',
         fg='red',
         font=('Arial',15),
         width=5,
         height=1).place(x=400,y=285)
l7_number=tk.Label(root,
         textvariable=number[7],
         font = ('Arial', 15),
         width = 5,
         fg='black',
         bg='#C0C0C0',
         height= 1 ).place(x=350,y=285)
number[7].set(7)

l8=tk.Label(root,
         textvariable=softmax_out_dot[8],
         bg='green',
         fg='red',
         font=('Arial',15),
         width=5,
         height=1).place(x=400,y=325)
l8_number=tk.Label(root,
         textvariable=number[8],
         font = ('Arial', 15),
         width = 5,
         fg='black',
         bg='#C0C0C0',
         height= 1 ).place(x=350,y=325)
number[8].set(8)

l9=tk.Label(root,
         textvariable=softmax_out_dot[9],
         bg='green',
         fg='red',
         font=('Arial',15),
         width=5,
         height=1).place(x=400,y=365)
l9_number=tk.Label(root,
         textvariable=number[9],
         font = ('Arial', 15),
         width = 5,
         fg='black',
         bg='#C0C0C0',
         height= 1 ).place(x=350,y=365)
number[9].set(9)

root.mainloop()