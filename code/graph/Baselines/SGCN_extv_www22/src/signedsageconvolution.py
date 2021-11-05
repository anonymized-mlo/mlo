"""Layer classes."""

import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_scatter import scatter_add, scatter_mean
from torch_geometric.utils import remove_self_loops, add_self_loops
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')
import copy
from param_parser import parameter_parser

args = parameter_parser()

edge_features = pd.read_csv(args.edge_fe_path).sort_values(by=['user1'])
user1 = edge_features.loc[:,['user1']].values
user2 = edge_features.loc[:,['user2']].values
edge_cols = edge_features.columns
scaler = StandardScaler()
scaler.fit(edge_features)
edge_features = scaler.transform(edge_features)
edge_features = pd.DataFrame(data=edge_features, columns = edge_cols)
edge_features['user1'] = user1
edge_features['user2'] = user2
edge_features = edge_features.set_index(['user1', 'user2'], drop=False)

pca_ = False


if(pca_):
    del edge_features['user1']
    del edge_features['user2']
    n_components = 8
    pca = PCA(n_components=n_components)
    edge_features = pca.fit_transform(edge_features)
    edge_cols = ['pca_col_'+str(x) for x in range(n_components)]
    edge_features = pd.DataFrame(data=edge_features, columns = edge_cols)
    edge_features['user1'] = user1
    edge_features['user2'] = user2
    edge_features = edge_features.set_index(['user1', 'user2'], drop=False)


node_features = pd.read_csv(args.features_path)
att_cols_email = [xe for xe in edge_features.columns if ('eatt' in xe)]
att_cols_email_features = [xe for xe in edge_features.columns if ('fatt' in xe)]
meta_edge_features = [xe for xe in edge_features.columns if (xe not in (att_cols_email + att_cols_email_features + ['user1', 'user2']))]
all_edge_features = att_cols_email + att_cols_email_features + meta_edge_features
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


m = np.ones(shape=(node_features.shape[0],node_features.shape[0]))

for x, y in zip(edge_features.user1,edge_features.user2):
    m[x][y] = 0
    m[x][x] = 0
    m[y][y] = 0


del edge_features['user1']
del edge_features['user2']

def uniform(size, tensor):
    """
    Uniform weight initialization.
    :param size: Size of the tensor.
    :param tensor: Tensor initialized.
    """
    stdv = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)

class ListModule(torch.nn.Module):
    """
    Abstract list layer class.
    """
    def __init__(self, *args):
        """
        Model initializing.
        """
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        """
        Getting the indexed layer.
        """
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        """
        Iterating on the layers.
        """
        return iter(self._modules.values())

    def __len__(self):
        """
        Number of layers.
        """
        return len(self._modules)

class SignedSAGEConvolution(torch.nn.Module):
    """
    Abstract Signed SAGE convolution class.
    :param in_channels: Number of features.
    :param out_channels: Number of filters.
    :param norm_embed: Normalize embedding -- boolean.
    :param bias: Add bias or no.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 norm=True,
                 norm_embed=True,
                 bias=True):
        super(SignedSAGEConvolution, self).__init__()

        self.args = parameter_parser()

        self.edge_feature_type = all_edge_features
        self.pca_ = pca_

        if(self.pca_):
            self.edge_feature_type = [x for x in edge_features.columns if 'user1' not in x and 'user2' not in x]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm = norm
        self.norm_embed = norm_embed
        self.weight = Parameter(torch.Tensor(self.in_channels, out_channels))
        self.multihead_attn = torch.nn.MultiheadAttention(embed_dim = node_features.shape[1],  dropout=.01, num_heads= 1)
        self.multihead_attn_dp = torch.nn.MultiheadAttention(embed_dim  = self.out_channels, dropout=0.01, num_heads = 1)
        self.multihead_attn_dn = torch.nn.MultiheadAttention(embed_dim  = self.out_channels, dropout=0.01, num_heads = 1)

        if(self.pca_):
            self.multihead_attn = torch.nn.MultiheadAttention(embed_dim = n_components,  dropout=.01, num_heads= 1)
            self.multihead_attn_d = torch.nn.MultiheadAttention(embed_dim  = self.out_channels, dropout=0.01, num_heads = 1)

        self.msk = torch.from_numpy(m).float().to(device)

        """
        True False-------------------------------
        """
        self.msk_flag_dp = True
        self.msk_flag_dn = True
        if(self.args.edge_features_incls):
            self.include_edge_features = True

        if(not self.args.edge_features_incls):
            self.include_edge_features = False


        if(self.args.attention_include):
            self.include_attention = True

        if(not self.args.attention_include):
            self.include_attention = False

        self.edges_pool_flag = True

        self.big_mask = torch.from_numpy(m).float().to(device) # Overall mask of edges

        if(self.include_attention and self.include_edge_features):
            self.weight = Parameter(torch.Tensor(self.in_channels +  4*len(self.edge_feature_type), out_channels))
            self.multihead_attn_a = torch.nn.MultiheadAttention(embed_dim  = self.weight.shape[1], dropout=0.01, num_heads = 1)

        if(self.include_attention and not self.include_edge_features):
            self.weight = Parameter(torch.Tensor(self.in_channels, out_channels))
            self.multihead_attn_a = torch.nn.MultiheadAttention(embed_dim  = self.weight.shape[1], dropout=0.01, num_heads = 1)

        if(not self.include_attention and self.include_edge_features):
            self.weight = Parameter(torch.Tensor(self.in_channels + 4*len(self.edge_feature_type), out_channels))
            self.multihead_attn_a = torch.nn.MultiheadAttention(embed_dim  = self.weight.shape[1], dropout=0.01, num_heads = 1)

        if(not self.include_attention and not self.include_edge_features):
            self.weight = Parameter(torch.Tensor(self.in_channels , out_channels)) # + node_features.shape[1]
            self.multihead_attn_a = torch.nn.MultiheadAttention(embed_dim  = self.weight.shape[1], dropout=0.01, num_heads = 1)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)


        edge_features_pooled_out = []
        edge_features_pooled_in = []

        for xe in set(edge_features.index.get_level_values(0)).union(set(edge_features.index.get_level_values(1))):
            tmp = edge_features.query("user1 == {}".format(xe))
            if(len(tmp) == 0):
                tmp.loc[0] = [0]*tmp.shape[1]
            edge_features_pooled_out.append(tmp.loc[:,self.edge_feature_type].max().values)


        for xe in set(edge_features.index.get_level_values(0)).union(set(edge_features.index.get_level_values(1))):
            tmp = edge_features.query("user2 == {}".format(xe))
            if(len(tmp) == 0):
                tmp.loc[0] = [0]*tmp.shape[1]
            edge_features_pooled_in.append(tmp.loc[:,self.edge_feature_type].max().values)

        self.edge_features_pooled_out = torch.from_numpy(np.array(edge_features_pooled_out)).float().to(device)
        self.edge_features_pooled_in = torch.from_numpy(np.array(edge_features_pooled_in)).float().to(device)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters.
        """
        size = self.weight.size(0)
        uniform(size, self.weight)
        uniform(size, self.bias)

    def __repr__(self):
        """
        Create formal string representation.
        """
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels, self.out_channels)

class SignedSAGEConvolutionBase(SignedSAGEConvolution):
    """
    Base Signed SAGE class for the first layer of the model.
    """
    def forward(self, x, edge_index):
        """
        Forward propagation pass with features an indices.
        :param x: Feature matrix. # number of nodes*features_size
        :param edge_index: Indices. # edges data
        """

        edge_index, _ = remove_self_loops(edge_index, None)
        row, col = edge_index #row, col are source and target nodes, from these, the values for the edge information can be referred to.

        """
        Node features can be pooled with attention here.
        """
        if self.norm:
            out = scatter_mean(x[col], row, dim=0, dim_size=x.size(0))

            if(self.include_attention):
                q = torch.reshape(x, (x.shape[0], 1, x.shape[1]))
                k = torch.reshape(x, (x.shape[0], 1, x.shape[1]))
                v = torch.reshape(x, (x.shape[0], 1, x.shape[1]))

                attn_output, attn_output_weights = self.multihead_attn(query=q, key=k, value=v, attn_mask = self.msk)
                attn_output = torch.reshape(attn_output, (x.shape[0], x.shape[1]))

                out = torch.cat((attn_output, x), 1)
            else:
                out = torch.cat((out, x), 1)

        else:
            out = scatter_add(x[col], row, dim=0, dim_size=x.size(0))

        if(self.include_edge_features):
            out = torch.cat((out, self.edge_features_pooled_out, self.edge_features_pooled_in), 1)
            out = torch.cat((out, self.edge_features_pooled_out, self.edge_features_pooled_in), 1)


        out = torch.matmul(out, self.weight)

        if self.bias is not None:
            out = out + self.bias
        if self.norm_embed:
            out = F.normalize(out, p=2, dim=-1)
        return out

class SignedSAGEConvolutionDeep(SignedSAGEConvolution):
    """
    Deep Signed SAGE class for multi-layer models.
    """
    def forward(self, x_1, x_2, edge_index_pos, edge_index_neg):
        """
        Forward propagation pass with features an indices.
        :param x_1: Features for left hand side vertices.
        :param x_2: Features for right hand side vertices.
        :param edge_index_pos: Positive indices.
        :param edge_index_neg: Negative indices.
        :return out: Abstract convolved features.
        """
        edge_index_pos, _ = remove_self_loops(edge_index_pos, None)
        edge_index_pos, _ = add_self_loops(edge_index_pos, num_nodes=x_1.size(0))
        edge_index_neg, _ = remove_self_loops(edge_index_neg, None)
        edge_index_neg, _ = add_self_loops(edge_index_neg, num_nodes=x_2.size(0))

        row_pos, col_pos = edge_index_pos
        row_neg, col_neg = edge_index_neg

        """
        self.h_pos.append(torch.tanh(self.positive_base_aggregator(self.X, positive_edges)))
        self.h_neg.append(torch.tanh(self.negative_base_aggregator(self.X, negative_edges)))
        for i in range(1, self.layers): #for every batch. when this is passed, it should lool at the indices and then compute aftr edge inclusion.
            self.h_pos.append(torch.tanh(self.positive_aggregators[i-1](self.h_pos[i-1], self.h_neg[i-1], positive_edges, negative_edges)))
        """

        if self.norm:

            out_1 = scatter_mean(x_1[col_pos], row_pos, dim=0, dim_size=x_1.size(0)) #have nodes' edges' mean values here. source and target should be identified here.

            """
            Self-attention b/w the node features
            """

            if (self.include_attention):
                poss_d = torch.unique(torch.tensor(torch.cat((row_pos , col_pos)), dtype=torch.long), sorted=True) #only positive

                qdp = torch.reshape(x_1[poss_d], (x_1[poss_d].shape[0], 1, x_1[poss_d].shape[1]))
                kdp = torch.reshape(x_1[poss_d], (x_1[poss_d].shape[0], 1, x_1[poss_d].shape[1]))
                vdp = torch.reshape(x_1[poss_d], (x_1[poss_d].shape[0], 1, x_1[poss_d].shape[1]))

                if (self.msk_flag_dp):
                    mp = np.ones(shape=(qdp.shape[0],qdp.shape[0]))
                    for x, y in zip(row_pos, col_pos): # mask for accomodating non-edges
                        mp[x][y] = 0
                        mp[x][x] = 0
                        mp[y][y] = 0
                    self.msk_dp = torch.from_numpy(mp).float().to(device)
                    self.msk_flag_dp = False

                attn_output_dp, attn_output_weights_d = self.multihead_attn_dp(query=qdp, key=kdp, value=vdp, attn_mask = self.msk_dp)
                attn_output_dp = torch.reshape(attn_output_dp, (x_1.shape[0], x_1.shape[1]))

            """
            For each node:
            Negative and positive edges, in and out, are pooled differently
            """

            if(self.include_edge_features and self.edges_pool_flag):
                t_index = [(x, y) for x, y in zip(row_pos.cpu().detach().numpy(), col_pos.cpu().detach().numpy()) if (x!=y)]
                t_edge_features_pos = edge_features.loc[t_index,:]

                self.t_edge_features_po = t_edge_features_pos.groupby(level=0).max() # po - positive out, group by index , index is selected in the row above
                missing_indexes_po = set(self.t_edge_features_po.index)^set(node_features.index)
                missing_indexes_info = np.zeros(shape=(len(missing_indexes_po), self.t_edge_features_po.shape[1]))
                t_df = pd.DataFrame(data=missing_indexes_info, index=list(missing_indexes_po), columns = self.t_edge_features_po.columns)
                t_df = t_df.rename_axis('user1')
                self.t_edge_features_po = pd.concat([self.t_edge_features_po, t_df]).sort_index().values # missing are initilised and rest are concatenated.
                self.t_edge_features_po = torch.from_numpy(self.t_edge_features_po).float().to(device)
                self.t_edge_features_po = F.normalize(self.t_edge_features_po, p=2, dim=-1)

                self.t_edge_features_pi = t_edge_features_pos.groupby(level=1).max() # pi - positive in
                missing_indexes_pi = set(self.t_edge_features_pi.index)^set(node_features.index)
                missing_indexes_info = np.zeros(shape=(len(missing_indexes_pi), self.t_edge_features_pi.shape[1]))
                t_df = pd.DataFrame(data=missing_indexes_info, index=list(missing_indexes_pi), columns = self.t_edge_features_pi.columns)
                t_df = t_df.rename_axis('user2')
                self.t_edge_features_pi = pd.concat([self.t_edge_features_pi, t_df]).sort_index().values
                self.t_edge_features_pi = torch.from_numpy(self.t_edge_features_pi).float().to(device)
                self.t_edge_features_pi = F.normalize(self.t_edge_features_pi, p=2, dim=-1)


            out_2 = scatter_mean(x_2[col_neg], row_neg, dim=0, dim_size=x_2.size(0))


            if (self.include_attention):
                poss_n = torch.unique(torch.tensor(torch.cat((row_neg , col_neg)), dtype=torch.long), sorted=True)

                qdn = torch.reshape(x_2[poss_n], (x_2[poss_n].shape[0], 1, x_2[poss_n].shape[1]))
                kdn = torch.reshape(x_2[poss_n], (x_2[poss_n].shape[0], 1, x_2[poss_n].shape[1]))
                vdn = torch.reshape(x_2[poss_n], (x_2[poss_n].shape[0], 1, x_2[poss_n].shape[1]))

                if (self.msk_flag_dn):
                    mn = np.ones(shape=(qdn.shape[0],qdn.shape[0]))
                    for x, y in zip(row_neg, col_neg):
                        mn[x][y] = 0
                        mn[x][x] = 0
                        mn[y][y] = 0
                    self.msk_dn = torch.from_numpy(mn).float().to(device)
                    self.msk_flag_dn = False

                attn_output_dn, attn_output_weights_d = self.multihead_attn_dn(query=qdn, key=kdn, value=vdn, attn_mask = self.msk_dn)
                attn_output_dn = torch.reshape(attn_output_dn, (x_1.shape[0], x_1.shape[1]))

            if(self.include_edge_features and self.edges_pool_flag):
                t_index = [(x, y) for x, y in zip(row_neg.cpu().detach().numpy(), col_neg.cpu().detach().numpy()) if (x!=y)]
                t_edge_features_neg = edge_features.loc[t_index,:] # po - positive out
                self.t_edge_features_no = t_edge_features_neg.groupby(level=0).max() # no - negative out
                missing_indexes_no = set(self.t_edge_features_no.index)^set(node_features.index)
                missing_indexes_info = np.zeros(shape=(len(missing_indexes_no), self.t_edge_features_no.shape[1]))
                t_df = pd.DataFrame(data=missing_indexes_info, index=list(missing_indexes_no), columns = self.t_edge_features_no.columns)
                t_df = t_df.rename_axis('user1')
                self.t_edge_features_no = pd.concat([self.t_edge_features_no, t_df]).sort_index().values
                self.t_edge_features_no = torch.from_numpy(self.t_edge_features_no).float().to(device)
                self.t_edge_features_no = F.normalize(self.t_edge_features_no, p=2, dim=-1)

                self.t_edge_features_ni = t_edge_features_neg.groupby(level=1).max() # no - negative out
                missing_indexes_ni = set(self.t_edge_features_ni.index)^set(node_features.index)
                missing_indexes_info = np.zeros(shape=(len(missing_indexes_ni), self.t_edge_features_ni.shape[1]))
                t_df = pd.DataFrame(data=missing_indexes_info, index=list(missing_indexes_ni), columns = self.t_edge_features_ni.columns)
                t_df = t_df.rename_axis('user2')
                self.t_edge_features_ni = pd.concat([self.t_edge_features_ni, t_df]).sort_index().values
                self.t_edge_features_ni = torch.from_numpy(self.t_edge_features_ni).float().to(device)
                self.t_edge_features_ni = F.normalize(self.t_edge_features_ni, p=2, dim=-1)

        else:
            out_1 = scatter_add(x_1[col_pos], row_pos, dim=0, dim_size=x_1.size(0))
            out_2 = scatter_add(x_2[col_neg], row_neg, dim=0, dim_size=x_2.size(0))

        if(not self.include_attention):
            out = torch.cat((out_1, out_2, x_1, x_2), 1)

        if(self.include_attention):
            out = torch.cat((attn_output_dp, attn_output_dn,  x_1, x_2), 1)

        if(self.include_edge_features and self.edges_pool_flag):
            out = torch.cat((out, self.t_edge_features_no , self.t_edge_features_ni, self.t_edge_features_po , self.t_edge_features_pi), 1)

        if(self.include_edge_features and not self.edges_pool_flag):
            out = torch.cat((out, self.edge_features_pooled_out, self.edge_features_pooled_in), 1)


        """
        Attention for everything
        Should only be conducted between the edges that are present. Get edge list and update the attention mask accordingly.
        """
        out = torch.matmul(out, self.weight)

        if self.bias is not None:
            out = out + self.bias
        if self.norm_embed:
            out = F.normalize(out, p=2, dim=-1)
        if (self.include_edge_features and self.edges_pool_flag and self.args.edge_features_incls):
            return out, self.t_edge_features_no, self.t_edge_features_ni, self.t_edge_features_po, self.t_edge_features_pi
        if (not (self.include_edge_features and self.edges_pool_flag and self.args.edge_features_incls)):
            return out
