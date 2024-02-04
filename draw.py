from tsne_torch import TorchTSNE as TSNE
import torch
import matplotlib.pyplot as plt
import numpy as np
def draw_tSNE_b(embed_aug,embed,biased):
    all_embed=embed_aug
    all_embed.append(embed)
    all_embed=torch.stack(all_embed)
    random_node_embed=all_embed[:,10,:]
    X_embed=TSNE().fit_transform(random_node_embed)
    origin=X_embed[-1,:]
    mean=np.mean(X_embed,axis=0)
    plt.scatter(X_embed[:,0],X_embed[:,1],label='Augmented sample')
    plt.scatter(origin[0],origin[1],c='r',marker='*',linewidths=2,s=50,label='Anchor sample')
    plt.scatter(mean[0],mean[1],c='g',marker='^',linewidths=2,s=50,label='Mean of Aug')
    plt.legend()
    if biased:
        plt.savefig("TSNE_b.png")
    else:
        plt.savefig("TSNE_u.png")
def sample_embed(embed,y,num=50):
    ret_X=[]
    ret_y=[]
    for label in torch.unique(y):
        index=torch.nonzero(y==label).squeeze()
        temp=torch.randperm(index.shape[0])
        index=index[temp[0:num]]
        ret_X.append(embed[index])
        ret_y.extend([label.item()]*num)
    ret_X=torch.cat(ret_X)
    ret_y=np.array(ret_y)
    return ret_X,ret_y

import os
def draw_tSNE(embed,y,dataset):
    #print(torch.sum(torch.isnan(embed).flatten()))
    plt.show()
    embed=torch.nn.functional.normalize(embed)
    y=y.squeeze()
    num=100
    X,y=sample_embed(embed,y,num=num)
    X=TSNE().fit_transform(X)
    plt.scatter(X[:,0],X[:,1],c=y,cmap='coolwarm')
    path=os.path.join('TSNES',dataset)
    path=path+'.svg'
    plt.savefig(path)
    
    
