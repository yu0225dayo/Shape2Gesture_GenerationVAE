import numpy as np

#入出力の手形状やパーツ・全体形状を可視化する

handinf=[0,1,2,3,4,18,
         0,5,6,7,19,
         0,8,9,10,20,
         0,11,12,13,21,
         0,14,15,16,17,22]

def drawpts(data,label,ax):
    cz=0
    ct=0
    co=0
    for i in range(len(data)):
        if int(label[i])==0:
            if cz==1:
                zero=np.vstack((zero,data[i]))
            else:
                zero=data[i]
                cz=1
        elif int(label[i])==1:
            if co==1:
                one=np.vstack((one,data[i]))
            else:
                one=data[i]
                co=1
        else:
            if ct==1:
                two=np.vstack((two,data[i]))
            else:
                two=data[i]
                ct=1
    
    x,y,z=zero[:,0],zero[:,1],zero[:,2]

    ax.scatter(x,y,z,c="green")#untach

    x1,y1,z1=one[:,0],one[:,1],one[:,2]

    ax.scatter(x1,y1,z1,c="blue")#右手

    x2,y2,z2=two[:,0],two[:,1],two[:,2]

    ax.scatter(x2,y2,z2,c="red")#左手

def draw_torchparts(data,label,ax):
    cz=0
    ct=0
    co=0
    for i in range(len(data)):
        if int(label[i])==0:
            if cz==1:
                zero=np.vstack((zero,data[i]))
            else:
                zero=data[i]
                cz=1
        elif int(label[i])==1:
            if co==1:
                one=np.vstack((one,data[i]))
            else:
                one=data[i]
                co=1
        else:
            if ct==1:
                two=np.vstack((two,data[i]))
            else:
                two=data[i]
                ct=1
    

    x1,y1,z1=one[:,0],one[:,1],one[:,2]

    ax.scatter(x1,y1,z1,c="blue")#右手

    x2,y2,z2=two[:,0],two[:,1],two[:,2]

    ax.scatter(x2,y2,z2,c="red")#左手


def drawparts(data,ax,parts):
    cp=0
    if parts=="left":
        color="red"
    elif parts=="right":
        color="blue"
    else:
        color="green"
    for i in range(len(data)):
        if cp==1:
            out=np.vstack((out,data[i]))
        else:
            out=data[i]
            cp=1
        
    x,y,z=out[:,0],out[:,1],out[:,2]

    ax.scatter(x,y,z,c=color,s=10)#untach

def drawchamfer(data,ax,color):
    cp=0
    for i in range(len(data)):
        if cp==1:
            out=np.vstack((out,data[i]))
        else:
            out=data[i]
            cp=1
        
    x,y,z=out[:,0],out[:,1],out[:,2]

    ax.scatter(x,y,z,c=color)#untach


def drawhand(hand,color,ax,mode=None):

    hx,hy,hz=hand[:,0],hand[:,1],hand[:,2]
    s=0
    if mode!=None:
        ax.scatter(hx,hy,hz,c="black")
    
    for i in range(4):
        if i == 0:
            for k in range(5):
                j=k
                x=np.array([hx[handinf[j]],hx[handinf[j+1]]])
                y=np.array([hy[handinf[j]],hy[handinf[j+1]]])
                z=np.array([hz[handinf[j]],hz[handinf[j+1]]])
                ax.plot(x,y,z,c=color)
            s+=6
        if i ==3: 
            for k in range(5):
                
                j=s+k
                x=np.array([hx[handinf[j]],hx[handinf[j+1]]])
                y=np.array([hy[handinf[j]],hy[handinf[j+1]]])
                z=np.array([hz[handinf[j]],hz[handinf[j+1]]])
                ax.plot(x,y,z,c=color)
            
        else:
            for k in range(4):
                j=s+k
                x=np.array([hx[handinf[j]],hx[handinf[j+1]]])
                y=np.array([hy[handinf[j]],hy[handinf[j+1]]])
                z=np.array([hz[handinf[j]],hz[handinf[j+1]]])
                ax.plot(x,y,z,c=color)
            s+=5


    hx,hy,hz=hand[:,0],hand[:,1],hand[:,2]
    
    s=0
    
    for i in range(4):
        if i == 0:
            for k in range(5):
                j=k
                x=np.array([hx[handinf[j]],hx[handinf[j+1]]])
                y=np.array([hy[handinf[j]],hy[handinf[j+1]]])
                z=np.array([hz[handinf[j]],hz[handinf[j+1]]])
                ax.plot(x,y,z,c=color)
            s+=6
        if i ==4: 
            for k in range(5):
                
                j=s+k
                x=np.array([hx[handinf[j]],hx[handinf[j+1]]])
                y=np.array([hy[handinf[j]],hy[handinf[j+1]]])
                z=np.array([hz[handinf[j]],hz[handinf[j+1]]])
                ax.plot(x,y,z,c=color)
            
        else:
            for k in range(4):
                j=s+k
                x=np.array([hx[handinf[j]],hx[handinf[j+1]]])
                y=np.array([hy[handinf[j]],hy[handinf[j+1]]])
                z=np.array([hz[handinf[j]],hz[handinf[j+1]]])
                ax.plot(x,y,z,c=color)
            s+=5