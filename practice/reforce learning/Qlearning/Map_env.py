import numpy as np
import tkinter as tk
import time
#np.random.seed(1)
UNIT=40     #pixel

class Map(tk.Tk,object):
    def __init__(self,
                 H=4,
                 W=4,
                 trap_num=2):
        super(Map,self).__init__()
        self.action_set=['U','D','L','R']
        self.action_num=len(self.action_set)
        self.title('Map_tkinter_build')
        self.Map_H=H
        self.Map_W=W
        self.trap_num=trap_num
        self.geometry('{0}x{1}'.format(self.Map_H*UNIT,self.Map_W*UNIT))
        self._build_Map_()
    def _build_Map_(self):
        self.canvas=tk.Canvas(self,bg='black',
                              height=self.Map_H*UNIT,
                              width=self.Map_W*UNIT)
        '''
        for col in range(0,self.Map_W*UNIT,UNIT):
            #x0 y0 x1 y1
            self.canvas.create_line(col,0,col,self.Map_H*UNIT)
        for row in range(0, self.Map_H * UNIT, UNIT):
            self.canvas.create_line(0, row,  self.Map_H * UNIT,row)
        '''
        self.Charactor,self.Ant=self._Charactor_genration_()
        self.canvas.pack()
    def step(self,action):
        s=self.canvas.coords(self.Ant)
        base_action=np.array([0,0])
        if action==0:
            if s[1]>UNIT:
                base_action[1]-=UNIT
        elif action == 1:
            if s[1]<(self.Map_H-1)*UNIT:
                base_action[1]+=UNIT
        elif action == 2:
            if s[0]<(self.Map_W-1)*UNIT:
                base_action[0]+=UNIT
        elif action == 3:
            if s[0]>UNIT:
                base_action[0]-=UNIT
        self.canvas.move(self.Ant,base_action[0],base_action[1])
        s_=self.canvas.coords(self.Ant)
        if s_==self.canvas.coords(self.Charactor[len(self.Charactor)-1]):
            reward=1
            done=True
        else:
            for i in range(len(self.Charactor)):
                if s_==self.canvas.coords(self.Charactor[i]):
                    reward = -10
                    done = True
                    break
                else:
                    reward = 0
                    done = False
        return  s_,reward,done
    def _reset_(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.Ant)
        Original = np.array([int(UNIT / 2), int(UNIT / 2)])

        center = Original + np.array([UNIT * int(self.archor[len(self.archor)-1] / self.Map_W),
                                      UNIT * int(self.archor[len(self.archor)-1] % self.Map_W)])

        self.Ant=(self.canvas.create_oval(
             center[0] - int(UNIT * 0.375), center[1] - int(UNIT * 0.375),
             center[0] + int(UNIT * 0.375), center[1] + int(UNIT * 0.375),
             fill='red'
        ))

        return self.Ant
    def Render(self):
        time.sleep(1)
        self.update()
    def _Charactor_genration_(self):
        # rectangle trap
        Original = np.array([int(UNIT / 2), int(UNIT / 2)])
        # trap//Self//Exit
        archor = np.random.choice(self.Map_H * self.Map_W, self.trap_num+2,replace=False)
        Charactor=[]
        self.archor=archor
        for i in range(len(archor)-1):
            center = Original + np.array([UNIT * int(archor[i] / self.Map_W),
                                          UNIT * int(archor[i] % self.Map_W)])
            if (i < len(archor) - 2):
                Charactor.append(self.canvas.create_rectangle(
                    center[0] - int(UNIT * 0.375), center[1] - int(UNIT * 0.375),
                    center[0] + int(UNIT * 0.375), center[1] + int(UNIT * 0.375),
                    fill='#424200'
                ))
            elif (i == len(archor) - 2):
                Charactor.append(self.canvas.create_rectangle(
                    center[0] - int(UNIT * 0.375), center[1] - int(UNIT * 0.375),
                    center[0] + int(UNIT * 0.375), center[1] + int(UNIT * 0.375),
                    fill='#2828FF'
                ))
        center = Original + np.array([UNIT * int(archor[len(archor) - 1] / self.Map_W),
                                      UNIT * int(archor[len(archor) - 1] % self.Map_W)])
        Ant=self.canvas.create_oval(
                center[0] - int(UNIT * 0.375), center[1] - int(UNIT * 0.375),
                center[0] + int(UNIT * 0.375), center[1] + int(UNIT * 0.375),
                fill='red'
            )
        return Charactor,Ant