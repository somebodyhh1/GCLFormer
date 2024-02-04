import numpy as np
import functools

from sklearn.metrics import f1_score,accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize, OneHotEncoder

def repeat(n_times):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            results = [f(*args, **kwargs) for _ in range(n_times)]
            statistics = {}
            for key in results[0].keys():
                values = [r[key] for r in results]
                statistics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values)}
            print_statistics(statistics, f.__name__)
            return statistics
        return wrapper
    return decorator


def prob_to_one_hot(y_pred):
    ret = np.zeros(y_pred.shape, np.bool_)
    indices = np.argmax(y_pred, axis=1)
    for i in range(y_pred.shape[0]):
        ret[i][indices[i]] = True
    return ret


def print_statistics(statistics, function_name):
    print(f'(E) | {function_name}:', end=' ')
    for i, key in enumerate(statistics.keys()):
        mean = statistics[key]['mean']
        std = statistics[key]['std']
        print(f'{key}={mean:.4f}+-{std:.4f}', end='')
        if i != len(statistics.keys()) - 1:
            print(',', end=' ')
        else:
            print()

def get_pred(embeddings, y):
    X = embeddings.detach().cpu().numpy()
    Y = y.detach().cpu().numpy()
    Y = Y.reshape(-1, 1)
    onehot_encoder = OneHotEncoder(categories='auto').fit(Y)
    Y = onehot_encoder.transform(Y).toarray().astype(np.bool_)

    X = normalize(X, norm='l2')
    logreg = LogisticRegression(solver='liblinear')
    c = 2.0 ** np.arange(-10, 10)
    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                        test_size=1 - 0.1,shuffle=False)

    clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
                       param_grid=dict(estimator__C=c), n_jobs=8, cv=5,
                       verbose=0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)
    return y_pred


def GCN_label_classification(model,optimizer,X,edge_index,y,ratio):
    n,_=X.shape
    index=list(range(n))
    index=np.random.permutation(index)
    boundary=(int)(ratio*n)
    train_index=index[0:boundary]
    test_index=index[boundary:]
    criterion=torch.nn.NLLLoss()
    print("shape==",y.shape)
    y=y.squeeze(1)
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        embed=model(X,edge_index)
        embed=torch.nn.functional.log_softmax(embed,dim=1)
        loss=criterion(embed[train_index],y[train_index])
        print("loss==",loss)
        
        loss.backward()
        optimizer.step()
    
    print(y[train_index])
    pred=torch.argmax(embed[test_index],dim=-1)
    y_test=y[test_index]
    print(pred)
    print(y_test)
    print(embed)
    print("pred shape==",pred.shape)
    acc=torch.sum(pred==y_test)/y_test.shape[0]
    print("acc==",acc)
    return acc

def label_classification(embeddings,dataset):
    X = embeddings.detach().cpu().numpy()
    y=dataset.label
    num_nodes,_=X.shape
    train_ratio,val_ratio,test_ratio=0.1,0.1,0.8
    train_num=(int)(train_ratio*num_nodes)
    val_num=(int)((train_ratio+val_ratio)*num_nodes)
    idx=np.arange(num_nodes)
    np.random.shuffle(idx)
    train_idx=idx[:train_num]
    valid_idx=idx[train_num:val_num]
    test_idx=idx[val_num:]
    
    Y = y.detach().cpu().numpy()
    Y = Y.reshape(-1, 1)
    onehot_encoder = OneHotEncoder(categories='auto').fit(Y)
    Y = onehot_encoder.transform(Y).toarray().astype(np.bool_)

    X = normalize(X, norm='l2')

    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                        test_size=1 - 0.1)

    logreg = LogisticRegression(solver='liblinear')
    c = 2.0 ** np.arange(-10, 10)

    clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
                       param_grid=dict(estimator__C=c), n_jobs=8, cv=5,
                       verbose=0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)
    y_pred = prob_to_one_hot(y_pred)

    micro = f1_score(y_test, y_pred, average="micro")
    macro = f1_score(y_test, y_pred, average="macro")
    
    y_test=y_test.astype(int)
    y_false=y_test-y_pred
    n,c=y_false.shape
    errors=[]
    for i in range(c):
        t=y_false[:,i]
        index=np.where(t==-1)
        t[index]=0
        errors.append(np.sum(t))

    return micro,macro,errors

def label_classification_val(embeddings,dataset):
    X = embeddings.detach().cpu().numpy()
    y=dataset.label

    
    Y = y.detach().cpu().numpy()
    Y = Y.reshape(-1, 1)
    onehot_encoder = OneHotEncoder(categories='auto').fit(Y)
    Y = onehot_encoder.transform(Y).toarray().astype(np.bool_)

    X = normalize(X, norm='l2')
    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                        test_size=1 - 0.1)
    X_valid,X_test, y_valid, y_test=train_test_split(X_test, y_test,
                                                        test_size=1 - 0.1)
    logreg = LogisticRegression(solver='liblinear')
    c = 2.0 ** np.arange(-10, 10)

    clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
                       param_grid=dict(estimator__C=c), n_jobs=8, cv=5,
                       verbose=0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_valid)
    y_pred = prob_to_one_hot(y_pred).astype(int)
    val_micro = f1_score(y_valid, y_pred, average="micro")
    
    y_pred = clf.predict_proba(X_test)
    y_pred = prob_to_one_hot(y_pred).astype(int)
    test_micro = f1_score(y_test, y_pred, average="micro")

    y_test=y_test.astype(int)
    y_false=y_test-y_pred
    n,c=y_false.shape
    errors=[]
    for i in range(c):
        t=y_false[:,i]
        index=np.where(t==-1)
        t[index]=0
        errors.append(np.sum(t))

    return val_micro,test_micro,errors

def label_classification_val1(embeddings,dataset):
    X = embeddings.detach().cpu().numpy()
    y=dataset.label

    
    Y = y.detach().cpu().numpy()
    Y = Y.reshape(-1, 1)
    onehot_encoder = OneHotEncoder(categories='auto').fit(Y)
    Y = onehot_encoder.transform(Y).toarray().astype(np.bool_)

    X = normalize(X, norm='l2')
    X_train,y_train,X_valid,y_valid,X_test,y_test=\
        X[dataset.train_idx],Y[dataset.train_idx],X[dataset.valid_idx],Y[dataset.valid_idx],X[dataset.test_idx],Y[dataset.test_idx]

    logreg = LogisticRegression(solver='liblinear')
    c = 2.0 ** np.arange(-10, 10)

    clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
                       param_grid=dict(estimator__C=c), n_jobs=8, cv=5,
                       verbose=0)
    clf.fit(X_train, y_train)

    
    y_pred = clf.predict_proba(X_valid)
    y_pred = prob_to_one_hot(y_pred).astype(int)
    val_micro = f1_score(y_valid, y_pred, average="micro")
    
    y_pred = clf.predict_proba(X_test)
    y_pred = prob_to_one_hot(y_pred).astype(int)
    test_micro = f1_score(y_test, y_pred, average="micro")

    y_test=y_test.astype(int)
    y_false=y_test-y_pred
    n,c=y_false.shape
    errors=[]
    for i in range(c):
        t=y_false[:,i]
        index=np.where(t==-1)
        t[index]=0
        errors.append(np.sum(t))

    return val_micro,test_micro,errors
import torch

def normaliz(x):
    return torch.nn.functional.normalize(x,dim=1)
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score

def get_dis_with_center(x,y,z1,z2):
    x=normaliz(x)
    z1=normaliz(z1)
    z2=normaliz(z2)
    y=y.squeeze()
    min_y=torch.min(y)
    y=y-min_y
    max_y=torch.max(y)
    mu=[]
    origin_x=x.clone()
    origin_y=y.clone()
    x=torch.cat([x,z1,z2],dim=0)
    y=torch.cat([y,y,y],dim=0)
    y = y.detach().cpu().numpy()
    x=torch.nn.functional.normalize(x)
    norms=[]
    for label in range(max_y+1):
        indice=np.where(y==label)[0]
        temp=x[indice,:]
        mu.append(torch.mean(temp,dim=0))
    x=origin_x
    y=origin_y
    Y=y.clone()
    y = y.detach().cpu().numpy()
    x=torch.nn.functional.normalize(x)
    W=torch.stack(mu)
    y_pred=torch.mm(x,W.T).detach().cpu().numpy()
    y_pred = prob_to_one_hot(y_pred)
    Y = Y.reshape(-1, 1)
    Y=Y.detach().cpu()
    onehot_encoder = OneHotEncoder(categories='auto').fit(Y)
    Y = onehot_encoder.transform(Y).toarray().astype(np.bool_)
    micro = f1_score(Y, y_pred, average="micro")

    cluster_y_pred=np.argmax(y_pred,axis=1)
    cluster_y=np.argmax(Y,axis=1)
    NMI=normalized_mutual_info_score(cluster_y_pred,cluster_y)
    ARI=adjusted_rand_score(cluster_y_pred,cluster_y)

    for label in range(max_y+1):
        indice=np.where(y==label)[0]
        temp=x[indice,:]
        temp1=temp[torch.randperm(temp.size(0))]
        temp=temp-temp1
        norm=torch.norm(temp,p=2,dim=1)
        norm=torch.mean(norm).item()
        norms.append(norm)
        
    delta=np.mean(norms) 
    mu=torch.stack(mu)
    mu_y=mu[y]
    temp=torch.mul(x,mu_y)
    sim_y=torch.sum(temp,dim=1)
    norm1=torch.norm(x,dim=1)
    norm2=torch.norm(mu_y,dim=1)
    sim_y=sim_y/norm1/norm2
    mu_i=torch.stack([torch.sum(mu,dim=0)]*(max_y+1))

    mu_i=(mu_i-mu)/(max_y)
    mu_i=mu_i[y]
    temp=torch.mul(x,mu_i)
    sim_i=torch.sum(temp,dim=1)
    norm1=torch.norm(x,dim=1)
    norm2=torch.norm(mu_i,dim=1)
    sim_i=sim_i/norm1/norm2
    sim_y=torch.mean(sim_y).item()
    sim_i=torch.mean(sim_i).item()
    return sim_y,sim_i,delta,micro,NMI,ARI
    
def classes_num(dataset_str):
    return {'cora': 7,
            'citeseer': 6,
            'pubmed': 3,
            'cs': 15,
            'physics': 5,
            'photo': 8,
            'comp': 10}[dataset_str]


import torch.optim as optim
def accuracy(y_pred, y_true):
    y_true = y_true.squeeze().long()
    preds = y_pred.max(1)[1].type_as(y_true)
    correct = preds.eq(y_true).double()
    correct = correct.sum().item()
    return correct / len(y_true)
def node_classification_evaluation(data,dataset):
    labels=dataset.label
    train_mask=dataset.train_idx
    val_mask=dataset.valid_idx
    test_mask=dataset.test_idx
    num_classes=torch.unique(labels).size(0)
    labels=labels.squeeze()
    lr_classifier = LogisticRegression(num_dim=data.shape[1],
                                       num_class=num_classes,
                                       dropout=0.9).to(data.device)
    finetune_optimizer = optim.Adam(params=lr_classifier.parameters(),
                                    lr=0.001,
                                    weight_decay=0)
    criterion = torch.nn.CrossEntropyLoss()
    best_val_acc = -1
    best_epoch = -1
    best_val_test_acc = -1
    for f_epoch in range(500):
        lr_classifier.train()
        out = lr_classifier(data)
        # print(out.shape)
        loss = criterion(out[train_mask], labels[train_mask])
        finetune_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lr_classifier.parameters(), max_norm=3)
        finetune_optimizer.step()
        with torch.no_grad():
            lr_classifier.eval()
            pred = lr_classifier(data)
            train_acc = accuracy(pred[train_mask], labels[train_mask])
            val_acc = accuracy(pred[val_mask], labels[val_mask])
            test_acc = accuracy(pred[test_mask], labels[test_mask])

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_epoch = f_epoch
                best_val_test_acc = test_acc
            # print("f_epoch", f_epoch, "train acc", train_acc, "val acc", val_acc, "test acc", test_acc)
    return test_acc, best_val_acc, best_val_test_acc

def cal_neighbor_sim(embed,edge_index):
    u,v=edge_index
    embed_u=embed[u]
    embed_v=embed[v]
    sim=torch.nn.functional.cosine_similarity(embed_u,embed_v)
    print("neighbor sim=",torch.mean(sim))