from sympy import true

import numpy as np
import pandas as pd
import time
#from enum import Enum

# Enviroment parameter
NUM_STATE=6 #1 dimention

ACTION=['left','right']
# How greedy
EPSILON=0.9
# learning rate
ALPHA=0.1
# reward decade
GAMMA=0.9
MAX_ROUND=13
FRESH_TIME=0.3

def B_QT(n_state,action):
    table=pd.DataFrame(np.zeros((n_state,len(action))),columns=action)
    return  table
def C_ACT(state,qt):
    S_action_set = qt.iloc[state,:]
    if np.random.uniform()>EPSILON or S_action_set.all()==0:
        action=np.random.choice(ACTION)
    else:
        action=S_action_set.argmax()
    return action
def get_E_feedback(S,A):
    if A=='right':
        if S==NUM_STATE-2:
            # find target
            S_='terminal'
            R=1
        else:
            S_=S+1
            R=0
    else:
        R=0
        if S==0:
            S_=S
        else:
            S_=S-1
    return S_,R
def Update_E(S,episde,step_c):
    envlist=['-']*(NUM_STATE-1)+['T']
    if S=='terminal':
        interact='Episode %s: total_steps=%s' %(episde+1,step_c)
        print  '\r{}'.format(interact)+'\n'
        time.sleep(2)
        print '\r\n'
    else:
        envlist[S]='o'
        interact=''.join(envlist)
        print '\r{}'.format(interact)+'\n'
        time.sleep(FRESH_TIME)
    #SxA=qt.iloc[]
def RL():
    # predict=>(greedy target() +Reward)-Predict
    QT=B_QT(NUM_STATE,ACTION)
    for EPSILON in range(MAX_ROUND):
        step_count=0
        S=0
        is_T=False
        Update_E(S,EPSILON,step_count)
        while not is_T:
            A=C_ACT(S,QT)
            S_,R=get_E_feedback(S,A)

            Q_P=QT.ix[S,A]
            if S_!='terminal':
                Qtarget=R+GAMMA*QT.iloc[S_,:].max()
            else:
                Qtarget=R
                is_T=true
            QT.ix[S,A]+=ALPHA*(Qtarget-Q_P)
            S=S_
            step_count +=1
            Update_E(S,EPSILON,step_count)
    return QT
# Q list about SXA

if __name__=="__main__":
    QT=RL()
    print  '\r\n QTABLE:\n'
    print QT
