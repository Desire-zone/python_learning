import tkinter as tk

root = tk.Tk()
root.geometry('500x400')
#
# path = tk.StringVar()
# path.set('1')
path='123'
e = tk.Entry(root)
e.pack()

predict = tk.StringVar()

def pre():
   var=e.get()
   predict.set(path)

l = tk.Label(root,textvariable=predict,bg='green',font=('Arial',25),width=5,height=5)
l.pack()
b1 = tk.Button(root,text="识别",command=pre,width=6,height=2)
b1.pack()

root.mainloop()
