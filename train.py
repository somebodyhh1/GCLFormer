import argparse
import sys
import os, random
import numpy as np
import torch
import yaml
from yaml import SafeLoader
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
from sklearn.neighbors import kneighbors_graph
from torch_geometric.utils import dropout_adj,homophily
import wandb
from logger import Logger
from dataset import load_dataset,load_dataset_all
from data_utils import load_fixed_splits, adj_mul, get_gpu_memory_map,get_direct_neighbors
from eval import evaluate, eval_acc, eval_rocauc, eval_f1
from parse import parse_method, parser_add_main_args
from evalCL import label_classification,node_classification_evaluation,cal_neighbor_sim,label_classification_val,label_classification_val1
import time
import yaml
from utils import mask_test_edges,get_roc_score_node
from draw import draw_tSNE,draw_tSNE_b
result=False
debug=True
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import matplotlib.pyplot as plt
def draw_mu_micro(mu_micro):
    size=len(mu_micro)
    x_axis=list(range(size))
    plt.plot(x_axis,mu_micro)
    plt.savefig("mu_micro.png")
def cal_mask_e(n,edge_index,prob=0.5):
    mask=torch.ones((n,n))
    print(edge_index)
    row,col=edge_index
    for i in range(n):
        t=torch.nonzero(row==i)
        size=t.shape[0]
        idx=np.array((list)(range(size)))
        idx=np.random.permutation(idx)
        want_size=(int)(prob*size)
        t=t[idx]
        t=t[0:want_size]
        t=col[t]
        mask[i,t]=0
    return mask

def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    t = x.clone()
    t[:, drop_mask] = 0
    return t
def get_pos_edge(adj,drop_prob,rb_order,n):
    adj=dropout_adj(adj,p=drop_prob)[0]
    adjs=[]
    adjs.append(adj)
    for i in range(rb_order - 1): # edge_index of high order adjacency
        adj = adj_mul(adj, adj, n)
        adjs.append(adj)
    return adjs

# NOTE: for consistent data splits, see data_utils.rand_train_test_idx
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    
def main():
    ### Parse args ###
    global result,debug
    parser = argparse.ArgumentParser(description='General Training Pipeline')
    parser_add_main_args(parser)
    args = parser.parse_args()
    if result==True:
        wandb.init()
        args.dataset=wandb.config.dataset
        args.task=wandb.config.task
        dummy=wandb.config.dummy
        if args.task=='node_classification':
            t=args.dataset+"_node"
        if args.task=='link_prediction':
            t=args.dataset+"_link"
        if args.task=='clustering':
            t=args.dataset+"_clu"
        config = yaml.load(open('config.yaml'), Loader=SafeLoader)
        if t in config:
            config=config[t]
        else:
            print("Existing")
            return
        drop_rate1=config['drop_rate1']
        drop_rate2=config['drop_rate2']
        args.lamda=config['lambda']
        args.lr=config['lr']
        args.weight_decay=config['wd']
        args.rb_order=config['rb_order']
        total_epoch=config['epoch']
        args.tau2=config['tau2']
        
    elif debug==False:
        wandb.init()
        args.lr=wandb.config.lr
        args.weight_decay=wandb.config.weight_decay
        args.hidden_channels=wandb.config.hidden_channels
        args.M=wandb.config.M
        args.lamda=wandb.config.lamda
        args.tau2=wandb.config.tau2
        args.num_heads=wandb.config.num_heads
        args.dropout=wandb.config.drop_out
        drop_rate1=wandb.config.drop_rate1
        drop_rate2=wandb.config.drop_rate2
        total_epoch=2000
    else:
        if args.task=='node_classification':
            t=args.dataset+"_node"
        if args.task=='link_prediction':
            t=args.dataset+"_link"
        if args.task=='clustering':
            t=args.dataset+"_clu"
        config = yaml.load(open('config.yaml'), Loader=SafeLoader)
        if t in config:
            config=config[t]
        else:
            print("Existing")
            return
        drop_rate1=config['drop_rate1']
        drop_rate2=config['drop_rate2']
        args.lamda=config['lambda']
        args.lr=config['lr']
        args.weight_decay=config['wd']
        args.rb_order=config['rb_order']
        total_epoch=config['epoch']
        args.tau2=config['tau2']
    
    print(args)
    fix_seed(args.seed)

    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    ### Load and preprocess data ###
    path=os.path.join(args.data_dir, args.dataset)
    dataset = load_dataset_all(path, args.dataset, args.sub_dataset)
    #dataset=load_dataset(args.data_dir,args.dataset,args.sub_dataset)
    if len(dataset.label.shape) == 1:
        dataset.label = dataset.label.unsqueeze(1)
    dataset.label = dataset.label.to(device)
    print("label==",dataset.label)
    ### Basic information of datasets ###
    n = dataset.graph['num_nodes']
    e = dataset.graph['edge_index'].shape[1]
    # infer the number of classes for non one-hot and one-hot labels
    c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
    d = dataset.graph['node_feat'].shape[1]
    print(f"dataset {args.dataset} | num nodes {n} | num edge {e} | num node feats {d} | num classes {c}")
    homo=homophily(dataset.graph['edge_index'],dataset.label)
    print("homophily==",homo)
    dataset.graph['edge_index'], dataset.graph['node_feat'] = \
        dataset.graph['edge_index'].to(device), dataset.graph['node_feat'].to(device)

    ### Load method ###
    model = parse_method(args, dataset,  args.hidden_channels, d, device)

    ### Performance metric (Acc, AUC, F1) ###
    if args.metric == 'rocauc':
        eval_func = eval_rocauc
    elif args.metric == 'f1':
        eval_func = eval_f1
    else:
        eval_func = eval_acc

    logger = Logger(args.runs, args)

    model.train()
    print('MODEL:', model)

    adj=dataset.graph['edge_index']

    adj, _ = remove_self_loops(adj)
    adj, _ = add_self_loops(adj, num_nodes=n)
    neighbor_path=os.path.join("neighbors",args.dataset)
    neighbor_path=os.path.join(neighbor_path,'neighbor.pt')
    if os.path.exists(neighbor_path):
        direct_neighbors=torch.load(neighbor_path)
    else:
        os.makedirs(os.path.join("neighbors",args.dataset))
        direct_neighbors=torch.tensor(get_direct_neighbors(adj,n)).to(device)
        torch.save(direct_neighbors,neighbor_path)
    from copy import deepcopy
    adj0=deepcopy(adj)
    if args.task=='link_prediction':
        link_path=os.path.join('links',args.dataset)
        link_path=os.path.join(link_path,'links.pt')
        if os.path.exists(link_path):
            adj_orig,adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false=torch.load(link_path)
        else:
            os.makedirs(os.path.join('links',args.dataset))
            t= mask_test_edges(adj,dataset.graph['num_nodes'],dataset.label)
            torch.save(t,link_path)
            adj_orig,adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = t
    else:
        adj_train=adj
    adj_train=adj_train.to(device)
    micros=[]
    mu_micros=[]
    ### Training loop ###
    best_val_micro=0
    best_test_micro=0
    best_roc_val,best_ap_val=0,0
    
    best_val_micro_r=0
    best_test_micro_r=0
    best_roc_val_r,best_ap_val_r=0,0
    
    NMIs=[]
    ARIs=[]
    roc_ap=[]
    print(model.parameters())
    
    more_lamda=3.0/total_epoch
    torch.cuda.empty_cache()
    mask=cal_mask_e(n,adj_train,1).to(device)
    adjs=get_pos_edge(adj_train,0,args.rb_order,n)
    for run in range(args.runs):
        optimizer = torch.optim.Adam(model.parameters(),weight_decay=args.weight_decay, lr=args.lr)
        for epoch in range(1,total_epoch+1):
            model.train()
            optimizer.zero_grad()
            x=dataset.graph['node_feat']
            x1=drop_feature(x,drop_rate1)
            x2=drop_feature(x,drop_rate2)
            adjs1=get_pos_edge(adj_train,drop_rate1,args.rb_order,n)
            adjs2=get_pos_edge(adj_train,drop_rate1,args.rb_order,n)
            out1, link_loss1_ = model(x1, adjs1, args.tau1)
            out2, link_loss2_ = model(x2, adjs2, args.tau1)

            link_loss_=link_loss1_
            link_loss_.extend(link_loss2_)
            link_loss_=sum(link_loss_) / len(link_loss_)
            #CLloss=model.loss(out1,out2)
            CLloss=model.loss(out1,out2,direct_neighbors,adjs[0],mask,tau=args.tau2)
            loss=CLloss-args.lamda*link_loss_
            #args.lamda+=more_lamda
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            homos=[]
            if not debug:
                wandb.log(res)
            if epoch %100==0:
                if args.task=='node_classification':
                    model.eval()
                    out0, link_loss0_ = model(x, adjs, args.tau1,nb_random_features=1000)
                    embed=out0.detach().cpu().numpy()
                    embed=torch.from_numpy(embed).to(device)
                    micro,macro,errors=label_classification(embed,dataset)
                    print(micro,macro)
                    res={'micro':micro,'macro':macro}
                    micros.append([micro,macro])
                    print(res)
                    with open('result.txt','a') as f:
                        f.write((str)(micro)+" ")
                    if debug == False:
                        wandb.log(res)
                if args.task=='link_prediction':
                    model.eval()
                    out0, link_loss0_ = model(x, adjs, args.tau1)
                    cal_neighbor_sim(out0,adjs[0])
                    cal_neighbor_sim(out0,adj0)
                    roc_curr_val, ap_curr_val = get_roc_score_node(emb=out0.detach().cpu().numpy(),
                                                                   edges_pos=val_edges,
                                                                   edges_neg=val_edges_false,
                                                                   adj=adj_orig,num_nodes=n)
                    roc_curr_test, ap_curr_test = get_roc_score_node(emb=out0.detach().cpu().numpy(),
                                                                     edges_pos=test_edges,
                                                                     edges_neg=test_edges_false,
                                                                     adj=adj_orig,num_nodes=n)

                    roc_ap.append([roc_curr_test,roc_curr_val,ap_curr_test,ap_curr_val])
                    res={'test_roc_epoch':roc_curr_test,
                                   'val_roc_test':roc_curr_val,
                                   'test_ap_epoch':ap_curr_test,
                                   'val_ap_epoch':ap_curr_val}
                    print(res)
                    if debug == False:
                        wandb.log(res)
                    if roc_curr_val > best_roc_val and ap_curr_val > best_ap_val:
                        best_roc_val = roc_curr_val
                        best_ap_val = ap_curr_val
                        best_roc_test = roc_curr_test
                        best_ap_test = ap_curr_test
                    if roc_curr_test > best_roc_val_r and ap_curr_test > best_ap_val_r:
                        best_roc_val_r=roc_curr_test
                        best_ap_val_r=ap_curr_test
                        best_roc_test_r=roc_curr_val
                        best_ap_test_r=ap_curr_val
                if args.task=='clustering':
                    from evalCL import get_dis_with_center
                    model.eval()
                    out0, link_loss0_ = model(x, adjs, args.tau1)
                    sim_y,sim_i,delta,micro_,NMI,ARI=get_dis_with_center(out0,dataset.label,out1,out2)
                    print(NMI,ARI)
                    NMIs.append(NMI)
                    ARIs.append(ARI)
                        
            print(f'Epoch: {epoch:02d}, '
                f'CLloss: {CLloss:.4f}, '
                f'Link_Loss: {link_loss_:4f}, ')

        if args.task=='node_classification':
            res={'micro':micro}
            print(res)
            if not debug:
                wandb.log(res)
            #draw_mu_micro(mu_micros)
            for t in micros:
                print(t)
            micros=np.array(micros)[:,0]
            max_micro=np.max(micros)
            max_index=np.argmax(micros)
            print(max_micro,max_index) # The maximization is used to determine the best epoch
            print(micro)
            if not debug:
                wandb.log({"max_micro":max_micro,"max_index":max_index})
        elif args.task=='link_prediction':

            res={'best_roc_test':best_roc_test,'best_ap_test':best_ap_test,'best_roc_test_r':best_roc_test_r,
                 'best_ap_test_r':best_ap_test_r}
            print(res)
            for t in roc_ap:
                print(t)
            if not debug:
                wandb.log(res)
        elif args.task=='clustering':
            max_NMI=np.max(NMIs)
            max_idx=np.argmax(NMIs)
            max_ARI=ARIs[max_idx]
            res={'max_NMI':max_NMI,'max_ARI':max_ARI,'max_idx':max_idx}
            print(max_NMI,NMI) # The maximization is used to determine the best epoch
            if not debug:
                wandb.log(res)
    with open('result.txt','a') as f:
        f.write("\n")
    #results = logger.print_statistics()
if __name__ == '__main__':
    import time
    if result:
        debug=False
    if debug:
        main()
    else:
        wandb.login()
        
        curPath = os.path.dirname(os.path.realpath(__file__))
        yaml_path = os.path.join(curPath, "sweep.yaml")
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = f.read()
        sweep_config = yaml.load(config,Loader=yaml.FullLoader)

        sweep_id=wandb.sweep(sweep_config,project='GraphTransformer_CiteSeer_node3')
        
        wandb.agent(sweep_id,main)