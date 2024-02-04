import math,os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import degree
from torch.nn import MarginRankingLoss
import time
from utils import sample_edges
BIG_CONSTANT = 1e8
def create_normal_projection_matrix(m,d,seed=0):
    torch.manual_seed(seed)
    rand=torch.randn((m, d))
    return rand
def create_projection_matrix1(m, d, sigma=1, seed=0, scaling=0):
    w=torch.randn((m, d)) * sigma
    return w
def create_projection_matrix(m, d, sigma=1, seed=0, scaling=0, struct_mode=False):
    nb_full_blocks = int(m/d)
    block_list = []
    current_seed = seed
    for _ in range(nb_full_blocks):
        torch.manual_seed(current_seed)
        if struct_mode:
            q = create_products_of_givens_rotations(d, current_seed)
        else:
            unstructured_block = torch.randn((d, d)) * sigma
            q, _ = torch.qr(unstructured_block)
            q = torch.t(q)
        block_list.append(q)
        current_seed += 1
    remaining_rows = m - nb_full_blocks * d
    if remaining_rows > 0:
        torch.manual_seed(current_seed)
        if struct_mode:
            q = create_products_of_givens_rotations(d, current_seed)
        else:
            unstructured_block = torch.randn((d, d)) * sigma
            q, _ = torch.qr(unstructured_block)
            q = torch.t(q)
        block_list.append(q[0:remaining_rows])
    final_matrix = torch.vstack(block_list)

    current_seed += 1
    torch.manual_seed(current_seed)
    if scaling == 0:
        multiplier = torch.norm(torch.randn((m, d)) * sigma, dim=1)
    elif scaling == 1:
        multiplier = torch.sqrt(torch.tensor(float(d))) * torch.ones(m)
    else:
        raise ValueError("Scaling must be one of {0, 1}. Was %s" % scaling)

    return torch.matmul(torch.diag(multiplier), final_matrix)

def create_products_of_givens_rotations(dim, seed):
    nb_givens_rotations = dim * int(math.ceil(math.log(float(dim))))
    q = np.eye(dim, dim)
    np.random.seed(seed)
    for _ in range(nb_givens_rotations):
        random_angle = math.pi * np.random.uniform()
        random_indices = np.random.choice(dim, 2)
        index_i = min(random_indices[0], random_indices[1])
        index_j = max(random_indices[0], random_indices[1])
        slice_i = q[index_i]
        slice_j = q[index_j]
        new_slice_i = math.cos(random_angle) * slice_i + math.cos(random_angle) * slice_j
        new_slice_j = -math.sin(random_angle) * slice_i + math.cos(random_angle) * slice_j
        q[index_i] = new_slice_i
        q[index_j] = new_slice_j
    return torch.tensor(q, dtype=torch.float32)

def relu_kernel_transformation(data, is_query, projection_matrix=None, numerical_stabilizer=0.001):
    del is_query
    if projection_matrix is None:
        return F.relu(data) + numerical_stabilizer
    else:
        ratio = 1.0 / torch.sqrt(
            torch.tensor(projection_matrix.shape[0], torch.float32)
        )
        data_dash = ratio * torch.einsum("bnhd,md->bnhm", data, projection_matrix)
        return F.relu(data_dash) + numerical_stabilizer

def softmax_kernel_transformation(data, is_query, projection_matrix=None, numerical_stabilizer=0.000001,sigma=1):
    sigma2=sigma*sigma
    data_normalizer = 1.0 / torch.sqrt(torch.sqrt(torch.tensor(data.shape[-1], dtype=torch.float32)))
    data = data_normalizer * data
    ratio = 1.0 / torch.sqrt(torch.tensor(projection_matrix.shape[0], dtype=torch.float32))
    data_dash = torch.einsum("bnhd,md->bnhm", data, projection_matrix)
    diag_data = torch.square(data)
    diag_data = torch.sum(diag_data, dim=len(data.shape)-1)
    diag_data = diag_data / 2.0 * sigma2
    diag_data = torch.unsqueeze(diag_data, dim=len(data.shape)-1)
    last_dims_t = len(data_dash.shape) - 1
    attention_dims_t = len(data_dash.shape) - 3
    
    if is_query:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.max(data_dash, dim=last_dims_t, keepdim=True)[0]) + numerical_stabilizer
        )
    else:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.max(torch.max(data_dash, dim=last_dims_t, keepdim=True)[0],
                    dim=attention_dims_t, keepdim=True)[0]) + numerical_stabilizer
        )

    return data_dash

def origin_kernel_transformation(data, is_query, projection_matrix=None, numerical_stabilizer=0.0001):
    #print("test==")
    data_normalizer = 1.0 / torch.sqrt(torch.sqrt(torch.tensor(data.shape[-1], dtype=torch.float32)))
    data = data_normalizer * data
    ratio = 1.0 / torch.sqrt(torch.tensor(projection_matrix.shape[0], dtype=torch.float32))
    data_dash = torch.einsum("bnhd,md->bnhm", data, projection_matrix)

    data_dash=torch.cat([torch.sin(data_dash),torch.cos(data_dash)],dim=len(data_dash.shape)-1) #[b n h 2m]
    #print("shape==",diag_data.shape,data_dash.shape)

    return ratio*data_dash + numerical_stabilizer

def numerator(qs, ks, vs,sigma):
    sigma2=sigma*sigma
    tqs=torch.pow(qs,1.0/sigma2)
    tks=torch.pow(ks,1.0/sigma2)
    kvs = torch.einsum("nbhm,nbhd->bhmd", tks, vs)
    return torch.einsum("nbhm,bhmd->nbhd", tqs, kvs)

def denominator(qs, ks,sigma):
    sigma2=sigma*sigma

    tks=torch.pow(ks,1.0/sigma2)
    tqs=torch.pow(qs,1.0/sigma2)

    all_ones = torch.ones([tks.shape[0]]).to(tqs.device)
    ks_sum = torch.einsum("nbhm,n->bhm", tks, all_ones) # ks_sum refers to O_k in the paper
    return torch.einsum("nbhm,bhm->nbh", tqs, ks_sum)

def numerator_gumbel(qs, ks, vs):
    kvs = torch.einsum("nbhkm,nbhd->bhkmd", ks, vs) # kvs refers to U_k in the paper
    return torch.einsum("nbhm,bhkmd->nbhkd", qs, kvs)

def denominator_gumbel(qs, ks):
    all_ones = torch.ones([ks.shape[0]]).to(qs.device)
    ks_sum = torch.einsum("nbhkm,n->bhkm", ks, all_ones) # ks_sum refers to O_k in the paper
    return torch.einsum("nbhm,bhkm->nbhk", qs, ks_sum)

def kernelized_softmax(query, key, value, kernel_transformation, projection_matrix=None,projection_matrix1=None, edge_index=None, tau=0.25, return_weight=True,sigma=1):
    '''
    fast computation of all-pair attentive aggregation with linear complexity
    input: query/key/value [B, N, H, D]
    return: updated node emb, attention weight (for computing edge loss)
    B = graph number (always equal to 1 in Node Classification), N = node number, H = head number,
    M = random feature dimension, D = hidden size
    '''
    query = query / math.sqrt(tau)
    key = key / math.sqrt(tau)
    query_prime = kernel_transformation(query, True, projection_matrix,sigma=sigma) # [B, N, H, M]
    key_prime = kernel_transformation(key, False, projection_matrix,sigma=sigma) # [B, N, H, M]
    query_prime = query_prime.permute(1, 0, 2, 3) # [N, B, H, M]
    key_prime = key_prime.permute(1, 0, 2, 3) # [N, B, H, M]
    value = value.permute(1, 0, 2, 3) # [N, B, H, D]
    # compute updated node emb, this step requires O(N)
    z_num = numerator(query_prime, key_prime, value,sigma)
    z_den = denominator(query_prime, key_prime,sigma)

    z_num = z_num.permute(1, 0, 2, 3)  # [B, N, H, D]
    z_den = z_den.permute(1, 0, 2)
    z_den = torch.unsqueeze(z_den, len(z_den.shape))
    z_output = z_num / z_den # [B, N, H, D]
    
    if return_weight: # query edge prob for computing edge-level reg loss, this step requires O(E)
        start, end = edge_index
        query_prime = kernel_transformation(query, True, projection_matrix1) # [B, N, H, M]
        key_prime = kernel_transformation(key, False, projection_matrix1) # [B, N, H, M]
        query_prime = query_prime.permute(1, 0, 2, 3) # [N, B, H, M]
        key_prime = key_prime.permute(1, 0, 2, 3) # [N, B, H, M]
        query_end, key_start = query_prime[end], key_prime[start] # [E, B, H, M]
        edge_attn_num = torch.einsum("ebhm,ebhm->ebh", query_end, key_start) # [E, B, H]
        edge_attn_num = edge_attn_num.permute(1, 0, 2) # [B, E, H]
        attn_normalizer = denominator(query_prime, key_prime,sigma) # [N, B, H]
        edge_attn_dem = attn_normalizer[end]  # [E, B, H]
        edge_attn_dem = edge_attn_dem.permute(1, 0, 2) # [B, E, H]
        A_weight = edge_attn_num / edge_attn_dem # [B, E, H]
        return z_output, A_weight

    else:
        return z_output

def kernelized_gumbel_softmax(query, key, value, kernel_transformation, projection_matrix=None, edge_index=None,
                                K=10, tau=0.25, return_weight=True):
    '''
    fast computation of all-pair attentive aggregation with linear complexity
    input: query/key/value [B, N, H, D]
    return: updated node emb, attention weight (for computing edge loss)
    B = graph number (always equal to 1 in Node Classification), N = node number, H = head number,
    M = random feature dimension, D = hidden size, K = number of Gumbel sampling
    '''
    query = query / math.sqrt(tau)
    key = key / math.sqrt(tau)
    query_prime = kernel_transformation(query, True, projection_matrix) # [B, N, H, M]
    key_prime = kernel_transformation(key, False, projection_matrix) # [B, N, H, M]
    query_prime = query_prime.permute(1, 0, 2, 3) # [N, B, H, M]
    key_prime = key_prime.permute(1, 0, 2, 3) # [N, B, H, M]
    value = value.permute(1, 0, 2, 3) # [N, B, H, D]

    # compute updated node emb, this step requires O(N)
    gumbels = (
        -torch.empty(key_prime.shape[:-1]+(K, ), memory_format=torch.legacy_contiguous_format).exponential_().log()
    ).to(query.device) / tau # [N, B, H, K]
    key_t_gumbel = key_prime.unsqueeze(3) * gumbels.exp().unsqueeze(4) # [N, B, H, K, M]
    z_num = numerator_gumbel(query_prime, key_t_gumbel, value) # [N, B, H, K, D]
    z_den = denominator_gumbel(query_prime, key_t_gumbel) # [N, B, H, K]

    z_num = z_num.permute(1, 0, 2, 3, 4) # [B, N, H, K, D]
    z_den = z_den.permute(1, 0, 2, 3) # [B, N, H, K]
    z_den = torch.unsqueeze(z_den, len(z_den.shape))
    z_output = torch.mean(z_num / z_den, dim=3) # [B, N, H, D]

    if return_weight: # query edge prob for computing edge-level reg loss, this step requires O(E)
        start, end = edge_index
        query_end, key_start = query_prime[end], key_prime[start] # [E, B, H, M]
        edge_attn_num = torch.einsum("ebhm,ebhm->ebh", query_end, key_start) # [E, B, H]
        edge_attn_num = edge_attn_num.permute(1, 0, 2) # [B, E, H]
        attn_normalizer = denominator(query_prime, key_prime) # [N, B, H]
        edge_attn_dem = attn_normalizer[end]  # [E, B, H]
        edge_attn_dem = edge_attn_dem.permute(1, 0, 2) # [B, E, H]
        A_weight = edge_attn_num / edge_attn_dem # [B, E, H]

        return z_output, A_weight

    else:
        return z_output

def add_conv_relational_bias(x, edge_index, b, trans='sigmoid'):
    '''
    compute updated result by the relational bias of input adjacency
    the implementation is similar to the Graph Convolution Network with a (shared) scalar weight for each edge
    '''
    row, col = edge_index
    d_in = degree(col, x.shape[1]).float()
    d_norm_in = (1. / d_in[col]).sqrt()
    d_out = degree(row, x.shape[1]).float()
    d_norm_out = (1. / d_out[row]).sqrt()
    conv_output = []
    for i in range(x.shape[2]):
        if trans == 'sigmoid':
            b_i = torch.sigmoid(b[i])
        elif trans == 'identity':
            b_i = b[i]
        else:
            raise NotImplementedError
        value = torch.ones_like(row) * b_i * d_norm_in * d_norm_out
        adj_i = SparseTensor(row=col, col=row, value=value, sparse_sizes=(x.shape[1], x.shape[1]))
        conv_output.append( matmul(adj_i, x[:, :, i]) )  # [B, N, D]
    conv_output = torch.stack(conv_output, dim=2) # [B, N, H, D]
    return conv_output

class NodeFormerConv(nn.Module):
    '''
    one layer of NodeFormer that attentive aggregates all nodes over a latent graph
    return: node embeddings for next layer, edge loss at this layer
    '''
    def __init__(self, in_channels, out_channels, num_heads, kernel_transformation=softmax_kernel_transformation,sigma=1, projection_matrix_type='a',
                 nb_random_features=10, use_gumbel=True, nb_gumbel_sample=10, rb_order=0, rb_trans='sigmoid', use_edge_loss=True):
        super(NodeFormerConv, self).__init__()
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        self.Wv = nn.Linear(in_channels, out_channels * num_heads)
        self.Wo = nn.Linear(out_channels * num_heads, out_channels)
        if rb_order >= 1:
            self.b = torch.nn.Parameter(torch.FloatTensor(rb_order, num_heads), requires_grad=True)
        self.sigma=sigma
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.kernel_transformation = kernel_transformation
        self.projection_matrix_type = projection_matrix_type
        self.nb_random_features = nb_random_features
        self.use_gumbel = use_gumbel
        self.nb_gumbel_sample = nb_gumbel_sample
        self.rb_order = rb_order
        self.rb_trans = rb_trans
        self.use_edge_loss = use_edge_loss

    def reset_parameters(self):
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        self.Wv.reset_parameters()
        self.Wo.reset_parameters()
        if self.rb_order >= 1:
            if self.rb_trans == 'sigmoid':
                torch.nn.init.constant_(self.b, 0.1)
            elif self.rb_trans == 'identity':
                torch.nn.init.constant_(self.b, 1.0)

    def forward(self, z, adjs, tau,edge,label,nb_random_features=-1):
        B, N = z.size(0), z.size(1)
        query = self.Wq(z).reshape(-1, N, self.num_heads, self.out_channels)
        key = self.Wk(z).reshape(-1, N, self.num_heads, self.out_channels)
        value = self.Wv(z).reshape(-1, N, self.num_heads, self.out_channels)
        
        #attn=torch.einsum('bnhd,bmhd->bnmh',query,key)
        #attn=torch.exp(attn)
        if nb_random_features==-1:
            nb_random_features=self.nb_random_features
        get_projection_matrix=create_projection_matrix
        
        if self.projection_matrix_type is None:
            projection_matrix = None
        else:
            dim = query.shape[-1]
            seed = int(round(time.time() * 1000000))

            projection_matrix = get_projection_matrix(
                nb_random_features, dim, seed=seed,sigma=self.sigma).to(query.device)
            projection_matrix1 = get_projection_matrix(
                30, dim, seed=seed,sigma=self.sigma).to(query.device)
        # compute all-pair message passing update and attn weight on input edges, requires O(N) or O(N + E)
        if self.use_gumbel and self.training:  # only using Gumbel noise for training
            z_next, weight = kernelized_gumbel_softmax(query,key,value,self.kernel_transformation,projection_matrix,edge,
                                                  self.nb_gumbel_sample, tau, self.use_edge_loss,sigma=self.sigma)
        else:
            z_next, weight = kernelized_softmax(query, key, value, self.kernel_transformation, projection_matrix, projection_matrix1, edge,
                                                tau, self.use_edge_loss,sigma=self.sigma)

        # compute update by relational bias of input adjacency, requires O(E)
        for i in range(self.rb_order):
            z_next += add_conv_relational_bias(value, adjs[i], self.b[i], self.rb_trans)
        # aggregate results of multiple heads
        z_next = self.Wo(z_next.flatten(-2, -1))
        if self.use_edge_loss: # compute edge regularization loss on input adjacency
            pos_idx=torch.nonzero(label==1).squeeze()
            edge_pos=edge[:,pos_idx]
            weight_pos=weight[:,pos_idx,:]
            row, col = edge_pos
            d_in = degree(col, query.shape[1]).float()
            d_norm = 1. / d_in[col]
            d_norm_ = d_norm.reshape(1, -1, 1).repeat(1, 1, weight_pos.shape[-1])
            link_loss = torch.mean(weight_pos.log() * d_norm_)
            return z_next, link_loss, torch.mean(weight,dim=-1).squeeze()
        else:
            return z_next
        

class NodeFormer(nn.Module):
    '''
    NodeFormer model implementation
    return: predicted node labels, a list of edge losses at every layer
    '''
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, num_heads=4, dropout=0.0,
                 kernel_transformation=softmax_kernel_transformation, nb_random_features=30, use_bn=True, use_gumbel=True,
                 use_residual=True, use_act=False, use_jk=False, nb_gumbel_sample=10, rb_order=0, rb_trans='sigmoid', use_edge_loss=True,sigma=1,clustering=False):
        super(NodeFormer, self).__init__()

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))
        self.bns = nn.ModuleList()
        self.bns.append(nn.LayerNorm(hidden_channels))
        for i in range(num_layers):
            self.convs.append(
                NodeFormerConv(hidden_channels, hidden_channels, num_heads=num_heads, kernel_transformation=kernel_transformation,
                              nb_random_features=nb_random_features, use_gumbel=use_gumbel, nb_gumbel_sample=nb_gumbel_sample,
                               rb_order=rb_order, rb_trans=rb_trans, use_edge_loss=use_edge_loss,sigma=sigma))
            self.bns.append(nn.LayerNorm(hidden_channels))

        if use_jk:
            self.fcs.append(nn.Linear(hidden_channels * num_layers + hidden_channels, out_channels))
        else:
            self.fcs.append(nn.Linear(hidden_channels, out_channels))
        self.fch=nn.Linear(hidden_channels, hidden_channels)
        self.fc1 = torch.nn.Linear(out_channels, out_channels)
        self.fc2 = torch.nn.Linear(out_channels, out_channels)
        self.dropout = dropout
        if clustering:
            self.activation = F.relu
        else:
            self.activation = F.elu
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.use_act = use_act
        self.use_jk = use_jk
        self.use_edge_loss = use_edge_loss
        self.clustering=clustering
        self.pre_adj,self.pre_label,self.pre_score=None,None,None
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()
   
    def get_QK(self,x):
        return self.convs[0].get_QK(x)
        
    def forward(self, x, adjs, tau=1.0,nb_random_features=-1,start_sample=False):
        x = x.unsqueeze(0) # [B, N, H, D], B=1 denotes number of graph
        layer_ = []
        link_loss_ = []
        z = self.fcs[0](x)
        if self.use_bn:
            z = self.bns[0](z)
        z = self.activation(z)
        z = F.dropout(z, p=self.dropout, training=self.training)
        layer_.append(z)
        n=x.shape[1]
        edge,label=sample_edges(adjs[0],n,self.pre_adj,self.pre_label,self.pre_score)
        for i, conv in enumerate(self.convs):
            if self.use_edge_loss:
                z, link_loss,weight = conv(z, adjs, tau,edge,label)
                link_loss_.append(link_loss)
            else:
                z = conv(z, adjs, tau,nb_random_features=nb_random_features)
            if self.use_residual:
                z += layer_[i]
            if self.use_bn:
                z = self.bns[i+1](z)
            if self.use_act:
                z = self.activation(z)
            z = F.dropout(z, p=self.dropout, training=self.training)
            layer_.append(z)
        if start_sample:
            self.pre_adj=edge
            self.pre_label=label
            self.pre_score=weight
        else:
            self.pre_adj,self.pre_label,self.pre_score=None,None,None
        if self.use_jk: # use jk connection for each layer
            z = torch.cat(layer_, dim=-1)
        if self.clustering:
            z=self.activation(z)
        x_out = self.fcs[-1](z).squeeze(0)
        if self.clustering:
            x_out=self.normalize(x_out)
        x_out=self.activation(x_out)
        if self.use_edge_loss:
            return x_out, link_loss_
        else:
            return x_out
    def normalize(self,z):
        return torch.nn.functional.normalize(z,dim=1)
        
    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)
        
    def sim(self, z1: torch.Tensor, z2: torch.Tensor,edge_index):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        t=torch.mm(z1, z2.t())
        #t[edge_index[:,0],edge_index[:,1]]=0
        return t

    def neighbor_loss(self,z1,z2,direct_neighbors,tau=0.5):
        f = lambda x: torch.exp(x / tau)
        _,num_neighbor=direct_neighbors.shape
        z1=F.normalize(z1)
        z2=F.normalize(z2)
        sims=[]
        temp=(list)(range(num_neighbor))
        temp=np.random.permutation(temp)
        for i in temp[0:10]:
            index=direct_neighbors[:,i]
            zt=z2[index]
            temp=torch.mul(z1,zt)
            sim=torch.sum(temp,dim=1)
            sims.append(sim)
        sims=torch.stack(sims,dim=1)
        return torch.mean(sims,dim=1)

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,direct_neighbors,edge_index,mask,tau=0.5):
        f = lambda x: torch.exp(x / tau)
        refl_sim = f(self.sim(z1, z1,edge_index))
        between_sim = f(self.sim(z1, z2,edge_index))
        if direct_neighbors != None:
            sim=self.neighbor_loss(z1,z2,direct_neighbors,tau)
            lamda=0.5
            diag=(1-lamda)*between_sim.diag()+lamda*sim
        else:
            diag=between_sim.diag()
        #print(torch.mean(between_sim.diag()),torch.mean(f(sim)))
        if mask !=None:
            between_sim=torch.mul(between_sim,mask)
            refl_sim=torch.mul(refl_sim,mask)
        return -torch.log(
            diag
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
    import math
    def RF_semi_loss(self,z1,z2,tau=0.5):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        dim=z1.shape[-1]
        seed = int(round(time.time() * 1000000))
        projection_matrix = create_projection_matrix(
                        1000, dim, seed=seed).to(z1.device)
        z1_prime=softmax_kernel_transformation_loss(z1,projection_matrix)
        z2_prime=softmax_kernel_transformation_loss(z2,projection_matrix)
        tau12=math.sqrt(tau)
        z1_prime/=tau12
        z2_prime/=tau12
        #num
        temp=torch.mul(z1_prime,z2_prime)
        num=torch.sum(temp,dim=1)
        #dem
        temp=z1_prime+z2_prime
        temp=torch.sum(temp,dim=0,keepdim=True)
        dem=torch.mm(z1_prime,temp.T).squeeze()
        return -torch.log(num/(dem))
        
        
    def loss(self, z1: torch.Tensor, z2: torch.Tensor, direct_neighbors,edge_index,mask,
             mean: bool = True, batch_size: int = 0, tau=0.5):
        h1 = self.projection(z1)
        h2 = self.projection(z2)
        #h1,h2=z1,z2
        if batch_size == 0:
            l1 = self.semi_loss(h1, h2, direct_neighbors,edge_index,mask,tau)
            l2 = self.semi_loss(h2, h1, direct_neighbors,edge_index,mask,tau)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)
        #print("z1,z2 nan",torch.isnan(z1).any(),torch.isnan(z2).any())
        #print("l1,l2 nan",torch.isnan(l1).any(),torch.isnan(l2).any())
        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret
    
    def get_sim(self,h1,h2):
        temp=torch.sum(torch.mul(h1,h2),dim=1)
        norm1=torch.norm(h1,dim=1)
        norm2=torch.norm(h2,dim=1)
        temp=temp/norm1/norm2
        return temp
        
    def rank_loss(self,z1,z2,margin=0.6):
        rankloss = nn.MarginRankingLoss(margin=margin,reduction='mean')
        h1 = self.projection(z1)
        h2 = self.projection(z2)
        h1,h2=z1,z2
        n,_=h2.shape
        index=np.array(list(range(n)))
        index=np.random.permutation(index)
        hn=h2[index]
        sim_p=self.get_sim(h1,h2)
        sim_n=self.get_sim(h1,hn)
        y=torch.tensor([1]*n).to(h1.device)
        loss=rankloss(sim_p,sim_n,y)
        return loss
    
    def loss_cal(self, x, x_aug):
        T = 0.2
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
        return loss
    
def softmax_kernel_transformation_loss(data,  projection_matrix, numerical_stabilizer=0.000001,sigma=1):
    m=projection_matrix.shape[0]
    sigma2=sigma*sigma
    norm=torch.norm(data,dim=1)
    norm_m=-1*sigma2*norm*norm/2
    norm_m=torch.exp(norm_m)
    norm_m=norm_m/math.sqrt(m)
    norm_m=norm_m.unsqueeze(1)
    tmp=torch.exp(torch.mm(data,projection_matrix.T))
    return norm_m*tmp