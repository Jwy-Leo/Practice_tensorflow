from Map_env import Map
from QLearn_table import QLeran_table
def update():
    for episode in range(1000):
        OB=env._reset_()
        print episode
        while True:
            env.Render()
            action=RL.choose_action(str(OB))
            OB_,R,D=env.step (action)
            print str(OB_)+'\n'
            RL.learn(str(OB),action,R,str(OB_))
            OB=OB_
            if D:
                break

    print("game over")
    env.destory()
if __name__=="__main__":
    env=Map(H=6,
            W=6,
            trap_num=3)
    RL=QLeran_table(actions=list(range(env.action_num)),learning_rate=1,e_greedy=0.95)
    env.after(1000,update)
    env.mainloop()
