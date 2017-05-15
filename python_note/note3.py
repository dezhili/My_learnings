# GUI的终极选择 : Tkinter
import tkinter as tk 

# app = tk.Tk()
# app.title("FishC Demo")

# theLabel = tk.Label(app, text='我的第一个窗口程序!')
# theLabel.pack()

# app.mainloop()  # 主事件循环




# class APP:
#     def __init__(self, master):
#         frame = tk.Frame(master)
#         frame.pack(side=LEFT, padx=20, pady=20)

#         self.hi_there = tk.Button(frame, text="打招呼", bg='black', fg='white', command=self.say_hi)
#         self.hi_there.pack()
        
#     def say_hi(self):
#         print("Hello everyone! My name is ldz")    

# root = tk.Tk()
# app = APP(root)

# root.mainloop()




from tkinter import *

root = Tk()

textLabel = Label(root, 
                  text="您所下载的影片含有未成年人限制的内容,\n请满18周岁再来观看",
                  justify=LEFT,
                  padx=10)
textLabel.pack(side=LEFT)

photo = PhotoImage(file="cat_400_500.jpg")
imgLabel = Label(root, image = photo)
imgLabel.pack(side=RIGHT)

mainloop()