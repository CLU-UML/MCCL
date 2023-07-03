from torch_geometric.data import DataLoader
metric_to_consider = ["add_average_degree_connectivity",
                      "add_avg_neighbor_deg",
                      "add_eigenvector_centrality_numpy",
                      "add_katz_centrality_numpy",
                    "avg_clustering",
                    "common_neighbors",
                    "deg_cent",
                    "degree",
                    "degree_assortativity_coefficient",
                    "density",
                    "group_degree_centrality",
                    "large_clique_size",
                    "len_local_bridges",
                    "len_min_edge_dominating_set",
                    "len_min_maximal_matching",
                    "len_min_weighted_dominating_set",
                    "len_min_weighted_vertex_cover",
                    "ramsey_R2",
                    "local_node_connectivity",
                    "mean_degree_mixing_matrix",
                    "nb_edges",
                    "nb_nodes",
                    "node_connectivity" ,
                    "resource_allocation_index", "treewidth_min_degree",
                    "add_closeness_centrality"]

metric_sort = {}
metric_dataloader = {}

D_add_average_degree_connectivity = []
D_add_avg_neighbor_deg = []
D_add_eigenvector_centrality_numpy = []
D_add_katz_centrality_numpy = []
D_avg_clustering = []
D_common_neighbors = []
D_deg_cent = []
D_degree = []
D_degree_assortativity_coefficient = []
D_density = []
D_group_degree_centrality = []
D_large_clique_size = []
D_len_local_bridges = []
D_len_min_edge_dominating_set = []
D_len_min_maximal_matching = []
D_len_min_weighted_dominating_set = []
D_len_min_weighted_vertex_cover = []
D_ramsey_R2 = []
D_local_node_connectivity = []
D_mean_degree_mixing_matrix = []
D_nb_edges = []
D_nb_nodes = []
D_node_connectivity = []
D_resource_allocation_index = []
D_treewidth_min_degree = []
D_add_closeness_centrality = []
D_random = []

add_average_degree_connectivity_dataloader = None
add_avg_neighbor_deg_dataloader = None
add_eigenvector_centrality_numpy_dataloader = None
add_katz_centrality_numpy_dataloader =None
avg_clustering_dataloader = None
common_neighbors_dataloader = None
deg_cent_dataloader = None
degree_dataloader = None
degree_assortativity_coefficient_dataloader = None
density_dataloader = None
group_degree_centrality_dataloader = None
large_clique_size_dataloader = None
len_local_bridges_dataloader = None
len_min_edge_dominating_set_dataloader = None
len_min_maximal_matching_dataloader = None
len_min_weighted_dominating_set_dataloader = None
len_min_weighted_vertex_cover_dataloader = None
ramsey_R2_dataloader = None
local_node_connectivity_dataloader = None
mean_degree_mixing_matrix_dataloader =None
nb_edges_dataloader = None
nb_nodes_dataloader = None
node_connectivity_dataloader = None
resource_allocation_index_dataloader = None
treewidth_min_degree_dataloader = None
add_closeness_centrality_dataloader = None
random_dataloader = None

#@profile
def sort_metric_dataset(df_train_metric,args):
    
    global metric_sort
    global metric_to_consider
    
    for m in metric_to_consider:
        if "A" in args.metric_order:
            metric_sort[m + "_A"] = df_train_metric.sort_values(by=m,ascending=True).index.tolist()
        
        if "D" in args.metric_order:
            metric_sort[m + "_D"] = df_train_metric.sort_values(by=m,ascending=False).index.tolist()
        
    
    if args.add_random:
        if "A" in args.metric_order:
            metric_sort["random_A"] = df_train_metric.sort_values(by="random",ascending=True).index.tolist()
        if "D" in args.metric_order:
            
            metric_sort["random_D"] = df_train_metric.sort_values(by="random",ascending=False).index.tolist()
        

def trim_dataset_as_per_competency(train_set, metric, c, order):
    if "A" == order:
            key = metric + "_A"
    if "D" == order:       
            key = metric + "_D"
   
    D = metric_sort[key]
    nb_examples = int(c * len(train_set))
    
    return [train_set[i] for i in D[:nb_examples]]



def revise_metric_dataloader_combined(epoch, bs, train_set, c,metrics):
    combined_examples_idx = []
    for m in metrics:
        D = metric_sort[m]
        nb_examples = int(c * len(train_set))
        combined_examples_idx.extend(D[:nb_examples])

    combined_examples_idx = list(set(combined_examples_idx))
    combined_examples = [train_set[i] for i in combined_examples_idx]
    print("epoch = {} , number of combined examples = {}".format(epoch, len(combined_examples)))
    return DataLoader(combined_examples, batch_size=bs, shuffle=True, pin_memory=True, num_workers=0)

def revise_metric_dataloader(bs, train_set, c, args):
    global metric_dataloader
    global metric_to_consider

    
    for m in metric_to_consider:
        if "A" in args.metric_order:
            
            metric_dataloader[m + "_A"] = DataLoader(trim_dataset_as_per_competency(train_set, m, c, "A"), batch_size=bs, shuffle=True, pin_memory=True, num_workers=0)
            
        if "D" in args.metric_order:
            
            metric_dataloader[m + "_D"] = DataLoader(trim_dataset_as_per_competency(train_set, m, c, "D"), batch_size=bs, shuffle=True, pin_memory=True, num_workers=0)
            
    
    if args.add_random:
        metric_dataloader["random" + "_A"] = DataLoader(trim_dataset_as_per_competency(train_set, "random", c, "A"), batch_size=bs, shuffle=True, pin_memory=True, num_workers=0)
        metric_dataloader["random" + "_D"] = DataLoader(trim_dataset_as_per_competency(train_set, "random", c, "D"), batch_size=bs, shuffle=True, pin_memory=True, num_workers=0)

        
def revise_metric_dict(args):

        
    asc_dict = {}
    desc_dict = {}
    q_dict = {}
    metric_dict = {}
    if args.use_k_means:
        if args.dataset == "pgr":
                
            pgr_k_means_metric = ["add_average_degree_connectivity", "add_eigenvector_centrality_numpy", "degree", "degree_assortativity_coefficient", "density",
                                  "len_local_bridges", "mean_degree_mixing_matrix", "node_connectivity", "treewidth_min_degree", "add_closeness_centrality"] 
            for m in pgr_k_means_metric:
                if "A" in args.metric_order:
                    asc_dict[m +"_A"] = metric_dataloader[m + "_A"]
                if "D" in args.metric_order:
                    desc_dict[m +"_D"] = metric_dataloader[m + "_D"]

                    
        elif args.dataset == "gdpr":
            omim_k_means_metric = ["add_avg_neighbor_deg", "add_average_degree_connectivity", "add_eigenvector_centrality_numpy", "large_clique_size", "degree_assortativity_coefficient",
                                  "density", "add_katz_centrality_numpy", "group_degree_centrality", "treewidth_min_degree", "add_closeness_centrality"]
            for m in omim_k_means_metric:
                if "A" in args.metric_order:
                    asc_dict[m +"_A"] = metric_dataloader[m + "_A"]
                if "D" in args.metric_order:
                    desc_dict[m +"_D"] = metric_dataloader[m + "_D"]

    else:
        
        for m in metric_to_consider:
            if "A" in args.metric_order:
                asc_dict[m +"_A"] = metric_dataloader[m + "_A"]
            if "D" in args.metric_order:
                desc_dict[m +"_D"] = metric_dataloader[m + "_D"]     
                
    if args.add_random:
        
            print("random added")
            if "A" in args.metric_order:
                asc_dict["random" +"_A"] = metric_dataloader["random" + "_A"]
            if "D" in args.metric_order:
                 desc_dict["random" +"_D"] = metric_dataloader["random" + "_D"] 

    metric_dict.update(asc_dict)
    metric_dict.update(desc_dict)
    metric_dict.update(q_dict)
                
    
    
    return metric_dict
