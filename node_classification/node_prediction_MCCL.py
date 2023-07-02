import os
os.system("pip install ogb")

import argparse
import torch
import torch.nn.functional as F
from torch import nn
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric.datasets import Planetoid
import pandas as pd
import sys
from sklearn.preprocessing import normalize
import numpy as np
from logger import Logger


metric_sort_train = {}
metric_to_consider = []

def set_metric_to_consider(args):
    global metric_to_consider
    if args.dataset == "arxiv":
        metric_to_consider = ["degree", "deg_cent", "density", "len_local_bridges", "add_avg_neighbor_deg",
                              "large_clique_size", "add_eigenvector_centrality_numpy", "degree_assortativity_coefficient", "ramsey_R2", "mean_degree_mixing_matrix"]
    
    
    if args.dataset == "cora":
        metric_to_consider = ['treewidth_min_degree', 'node_connectivity', 'len_min_weighted_dominating_set','add_eigenvector_centrality_numpy','degree_assortativity_coefficient','degree', 'add_closeness_centrality','add_avg_neighbor_deg', 'avg_clustering', 'add_average_degree_connectivity' ]

df_train_metric = None
rbf_dict = {}

def sort_metric_dataset(dff,args): # this also supports CCL from our spaced repetation paper 
    
    metric_sort = {}

    for m in metric_to_consider:
        if "A" in args.metric_order:
            print(dff.columns)
            metric_sort[m + "_A"] = dff.sort_values(by=m,ascending=True).index.tolist()
        
        if "D" in args.metric_order:
            metric_sort[m + "_D"] = dff.sort_values(by=m,ascending=False).index.tolist()

    
    if args.add_random:
        if "A" in args.metric_order:
            metric_sort["random_A"] = dff.sort_values(by="random",ascending=True).index.tolist()
        if "D" in args.metric_order:
            
            metric_sort["random_D"] = dff.sort_values(by="random",ascending=False).index.tolist()
            
    return metric_sort

def fix_negative_value_if_any(values):
    tmp = np.array(values)
    min_value = tmp.min()
    if min_value < 0:
        print("fixing negative values")
        tmp = tmp + (-1*min_value)
    return tmp

def load_train_metric(args):
    if args.dataset == "arxiv":
        df_train_metric = pd.read_csv("../indices/ogbn_arxiv_col_indices.csv")
    
    if args.dataset == "cora":
        df_train_metric = pd.read_csv("../indices/cora_col_indices_80_10_10.csv")
        print(df_train_metric.columns)
    df_train_metric = df_train_metric.fillna(0)
    print("using l2 norm")
    for c in df_train_metric.columns.tolist()[2:]:
            tmp = fix_negative_value_if_any(df_train_metric[c])
            df_train_metric[c] = tmp 
            df_train_metric[c] = normalize(df_train_metric[c][:,np.newaxis], axis=0)
            
    if args.add_random:
        print("random added")
        df_train_metric['random'] = np.random.rand(len(df_train_metric)).tolist()
        df_train_metric['random'] = normalize(df_train_metric['random'][:,np.newaxis], axis=0)

    return df_train_metric
# *****************************************************************************************************************************************************************************************************


def get_updated_idx(model, data, train_idx,args,c, epoch):
    global  df_train_metric
    global  metric_sort_train
    
    if df_train_metric is None:
        df_train_metric = load_train_metric(args)
        
    if len(metric_sort_train) == 0:
        metric_sort_train = sort_metric_dataset(df_train_metric,args)   
        
    metric_name = []
    metric_loss_value = []
    
    dff =  df_train_metric
    nb_example_v = int(c * len(dff)) 
    
    model.eval()
    metric_sort =  metric_sort_train
    for m in metric_sort:
        order = metric_sort[m]
        c_idx = order[:nb_example_v]

        if args.prioritizing_approach == "loss_based":
            # compute loss
            out = model(data.x, data.adj_t)[c_idx]
            loss = F.nll_loss(out, data.y.squeeze(1)[c_idx], reduction = "mean")
            metric_name.append(m)
            metric_loss_value.append(loss.detach().cpu().item())

        if args.prioritizing_approach == "index_based":
            # compute sum of each index
            complexity_scores = dff[m.replace("_A", "").replace("_D", "")].to_numpy()[c_idx]
            metric_name.append(m)
            metric_loss_value.append(np.sum(complexity_scores))

        
    if args.loss_creteria == "min":
        metric_loss_value = np.array(metric_loss_value)
        j_idx = np.argmin(metric_loss_value)
        j = metric_name[j_idx]
    elif args.loss_creteria == "max":
        metric_loss_value = np.array(metric_loss_value)
        j_idx = np.argmax(metric_loss_value)
        j = metric_name[j_idx]
    else:
        j = None
        
    # once the name of the index is found, we use it
    # to get examples from train dataset 
    print(epoch, j)
    nb_example_t = int(c * len(train_idx))
    updated_train_idx = metric_sort_train[j][:nb_example_t]
    updated_train_idx = torch.tensor(updated_train_idx)
    return updated_train_idx

class SAGE_no_feat(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, x):
        super(SAGE_no_feat, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.dropout = dropout
        self.add_feat = x

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, adj_t)
        
        return x.log_softmax(dim=-1)


class SAGE_w_feat(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, x, feat):
        super(SAGE_w_feat, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        self.h1_feat = nn.Linear(x.shape[-1], hidden_channels)
        
        self.convs.append(SAGEConv(hidden_channels+hidden_channels, out_channels))

        self.dropout = dropout
        self.add_feat = x

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
     
        
        add_feat = self.h1_feat(self.add_feat) # replaced self.h1_feat(x)  with  self.h1_feat(self.add_feat) 
        x = torch.cat([x, add_feat], dim = -1)
        
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)


batch_loss_history = {}
global_batch_counter = 0
loss_history = {}

def update_loss_history(train_idx, current_loss_tensor):
    global loss_history
    for key, clt in zip(train_idx, current_loss_tensor):
        key = key.item()
        if loss_history.get(key) is None:
            loss_history[key] = [clt]
        else:
            loss_history[key].append(clt)


def update_competency(t, T, c_0, p):
    term = pow(((1 - pow(c_0,p))*(t/T)) + pow(c_0,p), (1/p))
    return min([1,term])

def train(model, data, train_idx, optimizer, args, epoch):
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]

    b_loss = F.nll_loss(out, data.y.squeeze(1)[train_idx], reduction = "none")
    update_loss_history(train_idx, b_loss)
    loss = b_loss.mean()
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc



def main():

    
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--l2', type=float, default=0)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--feat', type=str, default = "yes")
    
    parser.add_argument('--dataset', type=str, default = "arxiv")
    
    # metric order and loss creteria for CMCL
    parser.add_argument('--prioritizing_approach', type=str, default="none", help="none, loss_based, index_based")
    parser.add_argument('--loss_creteria', type=str, default= 'max', help= "loss_creteria: min/max")
    parser.add_argument('--metric_order', type=str, default= "A", help= " choose from A/D")
    parser.add_argument('--add_random', type=bool, default= False, help= "T/F")
    
    args = parser.parse_args()
    set_metric_to_consider(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    root = "/data/nidhi/datasets"
    if args.dataset == "arxiv":
        dataset = PygNodePropPredDataset('ogbn-arxiv', root = root,
                                     transform=T.ToSparseTensor())

        data = dataset[0]
        data.adj_t = data.adj_t.to_symmetric()
        data = data.to(device)

        split_idx = dataset.get_idx_split()
        train_idx = split_idx['train'].to(device)

        
    if args.dataset == "cora":
        dataset = Planetoid(root='/tmp/Cora', name='Cora', split = "full", num_train_per_class = 380, num_val = 271, num_test = 271)
        data = dataset[0]
        data.y = data.y.reshape(-1,1)
        data = T.ToSparseTensor()(data)
        data.adj_t = data.adj_t.to_symmetric()
        
        # creating a new split in cora, which overwrites default split
        nodes = data.train_mask.size()[0]
        nodes = range(nodes)
        from sklearn.model_selection import train_test_split
        nodes_train, nodes_test, y_train, y_test = train_test_split(nodes, data.y.tolist(), stratify=data.y.tolist(), test_size=0.2, random_state=42)
        nodes_val, nodes_test, y_val, y_test = train_test_split(nodes_test, y_test, stratify=y_test, test_size=0.5, random_state=42)

        data.train_mask = torch.tensor([True if i in nodes_train else False for i in nodes]).bool()
        data.val_mask = torch.tensor([True if i in nodes_val else False for i in nodes]).bool()
        data.test_mask = torch.tensor([True if i in nodes_test else False for i in nodes]).bool()
        
        print(data.train_mask.sum(), data.val_mask.sum(), data.test_mask.sum())
        
        
        # As plantoid is different from arxiv, it does not have split_idx by default
        # so creating split_idx explicitly
        data = data.to(device)
        split_idx = {}
        split_idx["train"] = torch.nonzero(data.train_mask).reshape(-1)
        split_idx["valid"] = torch.nonzero(data.val_mask).reshape(-1)
        split_idx["test"] = torch.nonzero(data.test_mask).reshape(-1)
        train_idx = split_idx['train'].to(device)
        
    if args.feat == "yes":
        model = SAGE_w_feat(data.num_features, args.hidden_channels,
                         dataset.num_classes, args.num_layers,
                         args.dropout, data.x, args.feat).to(device)
    else:
        model = SAGE_no_feat(data.num_features, args.hidden_channels,
                         dataset.num_classes, args.num_layers,
                         args.dropout, data.x).to(device)   
        
    
    
    evaluator = Evaluator(name='ogbn-arxiv')
    
    logger = Logger(args.runs, args)
    sys.stdout = sys.stderr = open(logger.get_log_file(args), "w")
    print(args)
    c_0 = 0.01
   
    for run in range(args.runs):

        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay = args.l2)
        for epoch in range(1, 1 + args.epochs):
                
            c = c_0 if epoch == 1 else c
            train_idx = split_idx['train'].to(device)

            if args.prioritizing_approach != "none":
                train_idx = get_updated_idx(model, data, train_idx, args,c, epoch)

            else:
                train_idx = train_idx
                
            model.train()
            print("total number of examples: ", len(train_idx))
            loss = train(model, data, train_idx, optimizer, args,epoch)
            result = test(model, data, split_idx, evaluator)
            logger.add_result(run, result)

            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%')
                
            sys.stdout.flush()
            if args.prioritizing_approach != "none":
                c = update_competency(epoch, args.epochs, c_0, 2)
            else:
                c = None
            sys.stdout.flush()

                
        logger.print_statistics(run)
        out = model(data.x, data.adj_t)
        
        '''
        CODE BELOW SAVES THE PREDICTION FOR STATISTICAL SIGNIFICANCE TEST
        '''
    
        from scipy.special import softmax
        #logits = out[split_idx['train']].detach().cpu().numpy()
        #logits_softmax = softmax(logits, axis=1)
        #proba_train = np.max(logits_softmax, axis=1)
        #np.savetxt(f'/competence_based_multiview_CL/acl_23_multiview_ccl_gnn/node_prediction/train_{args.dataset}_{args.feat}.out', proba_train, delimiter=',')
        
        #logits = out[split_idx['valid']].detach().cpu().numpy()
        #logits_softmax = softmax(logits, axis=1)
        #proba_val = np.max(logits_softmax, axis=1)
        #np.savetxt(f'/competence_based_multiview_CL/acl_23_multiview_ccl_gnn/node_prediction/val_{args.dataset}_{args.feat}_{args.prioritizing_approach}.out', proba_val, delimiter=',') 
        
        logits = out[split_idx['test']].detach().cpu().numpy()
        logits_softmax = softmax(logits, axis=1)
        proba_test = np.max(logits_softmax, axis=1)
        np.savetxt(f'/competence_based_multiview_CL/acl_23_multiview_ccl_gnn/node_prediction/test_{args.dataset}_{args.feat}_{args.prioritizing_approach}.out', proba_test, delimiter=',') 
        
    logger.print_statistics()


if __name__ == "__main__":
    main()
