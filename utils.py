import numpy as np
import torch
import sys
import yaml
import random
import argparse
import scipy
import pickle as pkl
import scipy.sparse as sp
from copy import deepcopy
import time
from torch_geometric.utils import degree
from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error
def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def get_neg_edge(n,adj_pos):
    ep=adj_pos.shape[1]
    idx=(list)(range(n))
    seed = int(round(time.time())/100)
    np.random.seed(seed)
    idx=np.random.permutation(idx)
    row=torch.tensor(idx[0:ep])
    np.random.seed(seed+1)
    idx=np.random.permutation(idx)
    col=torch.tensor(idx[0:ep])
    adj_neg=torch.stack([row,col],dim=0).to(adj_pos.device)
    test_pos=torch.mul(adj_pos[0],adj_pos[1])
    test_neg=torch.mul(adj_neg[0],adj_neg[1])
    tensor_is_not_in = torch.isin(test_neg,test_pos,invert=True)
    adj_neg=adj_neg[:,tensor_is_not_in]
    return adj_neg
def sample_edges(adj,n,pre_adj,pre_label,pre_score,epsilon_l=0,epsilon_h=1):
    
    if pre_adj == None:
        adj_pos=adj
        adj_neg=get_neg_edge(n,adj_pos)
    else:
        pos_idx=torch.nonzero(pre_label==1).squeeze()
        neg_idx=torch.nonzero(pre_label==0).squeeze()
        pre_pos_score,pre_neg_score=pre_score[pos_idx],pre_score[neg_idx]
        pre_pos,pre_neg=pre_adj[:,pos_idx],pre_adj[:,neg_idx]
        avg_pos_attn=torch.mean(pre_pos_score)
        new_pos_idx=torch.nonzero(pre_neg_score>0.7).squeeze(1)
        print("avg score==",torch.mean(pre_pos_score),torch.mean(pre_neg_score),new_pos_idx.shape)
        new_pos=pre_neg[:,new_pos_idx]
        k=new_pos_idx.shape[0]
        _,idx=torch.topk(pre_pos_score,k,largest=False)
        mask=torch.ones_like(pos_idx)
        mask[idx]=False
        pre_pos_reserved=pre_pos[:,mask]
        adj_pos=torch.cat([pre_pos_reserved,new_pos],dim=-1)
        adj_neg=get_neg_edge(n,adj_pos)
        
    ep=adj_pos.shape[1]
    en=adj_neg.shape[1]
    labelp=torch.ones((ep,))
    labeln=torch.zeros((en,))
    label=torch.cat([labelp,labeln])
    adj=torch.cat([adj_pos,adj_neg],dim=1)
    
    return adj,label
    
def mask_test_edges(adj,num_nodes,label):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    label=label.squeeze()
    edges=adj
    row,col=edges
    idx=torch.nonzero(label[row]==label[col]).squeeze()
    adj=adj[:,idx]
    
    adj=adj.detach().cpu().numpy()
    data = np.ones(adj.shape[1])
    adj=sp.csr_matrix((data,(adj[0, :], adj[1, :])), shape=(num_nodes,num_nodes))
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    adj_orig=deepcopy(adj)
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]

    
    
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if label[idx_i]==label[idx_j]:
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    # assert ~ismember(test_edges_false, edges_all)
    # assert ~ismember(val_edges_false, edges_all)
    # assert ~ismember(val_edges, train_edges)
    # assert ~ismember(test_edges, train_edges)
    # assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    edge_train=deepcopy(train_edges.T)
    temp=np.array([train_edges[:, 1],train_edges[:, 0]])
    edge_train=torch.from_numpy(np.concatenate([edge_train,temp],axis=1))
    edge_train=edge_train.long()
    # NOTE: these edge lists only contain single direction of edge!
    return adj_orig, edge_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false

def get_new_edge_index(embed,e):
    def sigmoid(x):
        return 1 / (1 + torch.exp(-x))
    scores = torch.mm(embed, embed.T).flatten()
    _,idxs=torch.topk(scores,e)
    rows=[]
    cols=[]
    n,_=embed.shape
    for idx in idxs:
        idx=idx.item()
        rows.append((int)(np.floor(idx / n)))
        cols.append(idx % n)
    rows=torch.tensor(rows,dtype=torch.long)
    cols=torch.tensor(cols,dtype=torch.long)
    return torch.stack([rows,cols],dim=0)
    

def get_roc_score_node(edges_pos, edges_neg, emb, adj,num_nodes):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score