import torch
import numpy as np
import networkx as nx
import scipy.sparse as sp
import scipy.sparse.linalg as slinalg
from torch_geometric.utils import to_scipy_sparse_matrix, to_networkx
import collections
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
#from nltk.cluster.kmeans import KMeansClusterer
from myKmeans import KMeansClusterer
from nltk.cluster.util import cosine_distance
from gensim.models import Word2Vec
import copy
from random import sample

def getHighHighEdgeDic(subG):
    dic_node_deg = dict(subG.degree())
    edge_deg_dic = {}
    for edge in subG.edges():
        v1 = edge[0]
        v2 = edge[1]
        avg_deg = (dic_node_deg[v1]+dic_node_deg[v2])/2
        edge_deg_dic[edge] = avg_deg
    return edge_deg_dic

def getHighDegClassNodes(g, data, class_wise_train_idx_dic):
    temp_class_wise_train_idx_dic = copy.deepcopy(class_wise_train_idx_dic)
    for label in temp_class_wise_train_idx_dic:
        nodes_index_of_label_i = np.where(data.y == label)[0]
        g_sub = g.subgraph(nodes_index_of_label_i)
        high_high_edge_dic = getHighHighEdgeDic(g_sub)
        list_edge_deg_tple = sorted(high_high_edge_dic.items(), key = lambda kv:(kv[1], kv[0]), reverse = True)
        high_deg_edges = [k[0] for k in list_edge_deg_tple]
        if len(high_deg_edges) > 0:
            class_wise_train_idx_dic[label].extend([high_deg_edges[0][0], high_deg_edges[0][1]])
    return class_wise_train_idx_dic

# def absorption_probability(W, alpha=1e-6):
# 	n = W.shape[0]
# 	print('Calculate absorption probability...')
# 	W = W.copy().astype(np.float32)
# 	D = W.sum(1).flat
# 	L = sp.diags(D, dtype=np.float32) - W
# 	L += alpha * sp.eye(W.shape[0], dtype=L.dtype)
# 	L = sp.csc_matrix(L)
# 	A = slinalg.inv(L).toarray()
# 	print('Calculate absorption probability...DONE')
# 	return A#(L + L*L).toarray()

def absorption_probability(W, alpha=1e-6):
	n = W.shape[0]
	print('Calculate absorption probability...')
	W = W.copy().astype(np.float32)
	D = W.sum(1).flat
	L = sp.diags(D, dtype=np.float32) - W
	L += alpha * sp.eye(W.shape[0], dtype=L.dtype)
	L = sp.csc_matrix(L)
	A = slinalg.inv(L).toarray()
	#A = (1.0/alpha)*(sp.eye(W.shape[0], dtype=L.dtype) - (1.0/alpha)*L + (1.0/alpha)*(1.0/alpha)*L*L)
	print('Calculate absorption probability...DONE')
	return sp.csc_matrix(A)

def applyKnnKmeans(data, index):
    node_features = data.x[index].numpy().copy()
    if node_features.shape[0] > 1:
        kmeans = KMeansClusterer(num_means=2, distance=cosine_distance, repeats=25, avoid_empty_clusters=True)
        kmeans_labels_ = np.array(kmeans.cluster(node_features, assign_clusters=True))
        elements_count = collections.Counter(kmeans_labels_)
        larger_label = 0
        larger_freq = elements_count[0]
        c1 = np.where(kmeans_labels_ == 0); c2 = np.where(kmeans_labels_ == 1)
        c1_center = kmeans.means()[0]; c2_center = kmeans.means()[1]
        if elements_count[0] < elements_count[1]:#if label 0 has less frequency than label 1
            c1, c2 = c2, c1 #exchange the indices of the labels
            c1_center, c2_center = c2_center, c1_center #exchange the centers
            larger_label = 1
            larger_freq = elements_count[1]

        index_prime = np.setdiff1d(np.array([nd for nd in range(data.num_nodes)]), index)
        index_prime_features = data.x[index_prime].numpy().copy()
        c1_indexPrime_distance = np.array([cosine_similarity(np.array([c1_center]), np.array([ipf])) for ipf in index_prime_features])
        res = sorted(sorted(range(len(c1_indexPrime_distance)), key = lambda sub: c1_indexPrime_distance[sub], reverse=True)[:larger_freq])
        c1_center_nearest_nbs = index_prime[res]
        new_index = np.union1d(c1_center_nearest_nbs, index[c2])
    else:
        new_index = index

    return list(new_index)

def print_error_rate(Y, c, index):
    count = 0
    for i in index:
        if Y[i] != c:
            count += 1
    print(count/len(index))

def getParwalk(adj, class_wise_train_idx_dic, data, k=3):
	num_nodes = data.num_nodes
	A = absorption_probability(adj, alpha=1e-6)
	laplacian = sp.diags(adj.sum(1).flat, 0) - adj
	laplacian = laplacian.astype(np.float32).tocoo()
	eta = adj.shape[0]/(adj.sum()/adj.shape[0])**len('cc')
	train_size = np.array([len(class_wise_train_idx_dic[class_label]) for class_label in class_wise_train_idx_dic])

	model_config_t = (train_size*k*eta/train_size.sum()).astype(np.int64)
	G  = nx.to_networkx_graph(adj)
	class_wise_train_idx_dic = getHighDegClassNodes(G, data, class_wise_train_idx_dic)

	all_indices = []
	changed_labels = copy.copy(data.y)
	for label in class_wise_train_idx_dic:
		oneHotVec = np.array([0 for i in range(num_nodes)])
		oneHotVec[class_wise_train_idx_dic[label]] = 1
		a = A.dot(oneHotVec)
		gate = (-np.sort(-a, axis=0))[model_config_t[label]]
		index = np.where(a.flat > gate)[0]
		#print(model_config_t, index)
		#Now apply kmeans and knn
		new_indices = applyKnnKmeans(data, index)
		#print(index, new_indices)
		all_indices += new_indices + class_wise_train_idx_dic[label]
		print_error_rate(data.y, label, new_indices)
		changed_labels[new_indices] = label
		#print(all_indices)
	return all_indices, changed_labels


def ipLevelIntervention_pwkk(dataset, data, class_wise_train_idx_dic, args):
	k = 3
	if args.dataset in ['computers', 'photo']:
	    k = 50
	adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=len(data.y))
	all_indices, changed_labels = getParwalk(adj, class_wise_train_idx_dic, data, k=k)
	return torch.LongTensor(all_indices), changed_labels


def get_randomwalk(G, node, path_length):
    import random
    random_walk = [str(node)]

    for i in range(path_length-1):
        temp = list(G.neighbors(node))
        temp = list(set(temp) - set(random_walk))
        if len(temp) == 0:
            break

        random_node = random.choice(temp)
        random_walk.append(str(random_node))
        node = random_node

    return random_walk

def getDeepWalkModel(adj, data, class_wise_train_idx_dic=None):
    G  = nx.to_networkx_graph(adj)
    if class_wise_train_idx_dic != None:
        class_wise_train_idx_dic = getHighDegClassNodes(G, data, class_wise_train_idx_dic)
    random_walks = []
    for n in G.nodes():
        for i in range(len(class_wise_train_idx_dic)):
            random_walks.append(get_randomwalk(G, n, 2))

    model = Word2Vec(window = 2, sg = 1, hs = 16,  negative = 10,  alpha=0.03, min_alpha=0.0007, seed = 14)
    model.build_vocab(random_walks, progress_per=2)
    model.train(random_walks, total_examples = model.corpus_count, epochs=20, report_delay=1)

    return model

def getDeepWalkIPI(adj, class_wise_train_idx_dic, data, k=3):
    laplacian = sp.diags(adj.sum(1).flat, 0) - adj
    laplacian = laplacian.astype(np.float32).tocoo()
    eta = adj.shape[0]/(adj.sum()/adj.shape[0])**len('cc')
    train_size = np.array([len(class_wise_train_idx_dic[class_label]) for class_label in class_wise_train_idx_dic])
    model_config_t = (train_size*k*eta/train_size.sum()).astype(np.int64)

    deep_model = getDeepWalkModel(adj, data, class_wise_train_idx_dic)

    all_indices = []
    changed_labels = copy.copy(data.y)

    for label in class_wise_train_idx_dic:
        if model_config_t[label]//len(class_wise_train_idx_dic[label]) > 1:
            num_of_can = model_config_t[label]//len(class_wise_train_idx_dic[label])
        else:
            num_of_can = 1

        index = np.array([], dtype = int)

        for node in class_wise_train_idx_dic[label]:
            if str(node) in deep_model.wv:
                similar_nodes = deep_model.wv.similar_by_word(str(node),topn=num_of_can)
                int_similar_nodes = np.array(list(set([int(tup[0]) for tup in similar_nodes])))
                index = np.concatenate((index, int_similar_nodes), axis=0)
            else:
                index = np.concatenate((index, np.array([node])), axis=0)
	    #Now apply kmeans and knn

        new_indices = applyKnnKmeans(data, index)
        all_indices += new_indices + class_wise_train_idx_dic[label]
        changed_labels[index] = label

    return all_indices, changed_labels


def ipLevelIntervention_dwkk(dataset, data, class_wise_train_idx_dic, args):
    k = 3
    if args.dataset in ['computers', 'photo']:
        k = 50
    adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=len(data.y))
    all_indices, changed_labels = getDeepWalkIPI(adj, class_wise_train_idx_dic, data, k=k)
    return torch.LongTensor(all_indices), changed_labels


def getRandomSample(adj, class_wise_train_idx_dic, data, k=3):
    num_nodes = data.num_nodes
    eta = adj.shape[0]/(adj.sum()/adj.shape[0])**len('cc')
    train_size = np.array([len(class_wise_train_idx_dic[class_label]) for class_label in class_wise_train_idx_dic])

    model_config_t = (train_size*k*eta/train_size.sum()).astype(np.int64)
    G  = nx.to_networkx_graph(adj)

    all_indices = []
    changed_labels = copy.copy(data.y)
    all_nodes = [i for i in range(len(data.y))]
    for label in class_wise_train_idx_dic:
        # find a set of random nodes
        random_nodes = sample(all_nodes, model_config_t[label])
        #relabel their original label
        changed_labels[random_nodes] = label
        #add the training nodes
        all_indices += random_nodes
    return torch.LongTensor(all_indices), changed_labels


def ipLevelIntervention_ransample(dataset, data, class_wise_train_idx_dic, args):
    k = 3
    if args.dataset in ['computers', 'photo']:
        k = 50
    adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=len(data.y))
    all_indices, changed_labels = getRandomSample(adj, class_wise_train_idx_dic, data, k=k)
    return torch.LongTensor(all_indices), changed_labels

def getMetaPathSample(G, seed_nodes, num_nodes):
    collected_nodes = []
    while (len(collected_nodes) < num_nodes):
        updated_seed_nodes = []
        for seed in seed_nodes:
            #get the neighbors
            nbd_list = list(G.neighbors(seed))
            #get the degree of the neighbors
            nbd_deg_list = np.array([tup[1] for tup in G.degree(nbd_list)])
            #get the index of max deg
            max_deg_index = np.argmax(nbd_deg_list)
            #get the max deg neighbor
            max_deg_nbd = nbd_list[max_deg_index]
            #append the max_deg_nbd to the collected_nodes
            collected_nodes.append(max_deg_nbd)
            updated_seed_nodes.append(max_deg_nbd)
        # update the seed_nodes list
        seed_nodes = copy.copy(updated_seed_nodes)
    return list(set(collected_nodes))

def getMetaPathSample_V2(G, seed_nodes, num_nodes):
    collected_nodes = []
    while (len(collected_nodes) < num_nodes):
        updated_seed_nodes = []
        for seed in seed_nodes:
            #get the neighbors
            nbd_list = np.setdiff1d(np.array(list(G.neighbors(seed))), np.array(collected_nodes))
            if len(nbd_list) == 0:
                break
            #get the degree of the neighbors
            nbd_deg_list = np.array([tup[1] for tup in G.degree(nbd_list)])
            #get the index of max deg
            max_deg_index = np.argmax(nbd_deg_list)
            #get the max deg neighbor
            max_deg_nbd = nbd_list[max_deg_index]
            #append the max_deg_nbd to the collected_nodes
            collected_nodes.append(max_deg_nbd)
            updated_seed_nodes.append(max_deg_nbd)
        if len(set(seed_nodes) - set(updated_seed_nodes)) == 0:
            break
        # update the seed_nodes list
        seed_nodes = copy.copy(updated_seed_nodes)
    return list(set(collected_nodes))


def getMetaSample(adj, class_wise_train_idx_dic, data, k=3):
    num_nodes = data.num_nodes
    eta = adj.shape[0]/(adj.sum()/adj.shape[0])**len('cc')
    train_size = np.array([len(class_wise_train_idx_dic[class_label]) for class_label in class_wise_train_idx_dic])

    model_config_t = (train_size*k*eta/train_size.sum()).astype(np.int64)
    G  = nx.to_networkx_graph(adj)
    all_indices = []
    changed_labels = copy.copy(data.y)
    all_nodes = [i for i in range(len(data.y))]

    for label in class_wise_train_idx_dic:
        #find a meta path like set of nodes: high deg -- high deg ...
        mata_high_deg_nodes = getMetaPathSample(G, class_wise_train_idx_dic[label], model_config_t[label])
        #relabel their original label
        changed_labels[mata_high_deg_nodes] = label
        #add the training nodes
        all_indices += mata_high_deg_nodes
    return torch.LongTensor(all_indices), changed_labels


def ipLevelIntervention_metaPath(dataset, data, class_wise_train_idx_dic, args):
    k = 3
    if args.dataset in ['computers', 'photo']:
        k = 50
    adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=len(data.y))
    all_indices, changed_labels = getMetaSample(adj, class_wise_train_idx_dic, data, k=k)
    return torch.LongTensor(all_indices), changed_labels


################################## Output level Intervention ########################################
def changePredParwalkHighNbdOrStartAllClassThresh(A, g, pred, prediction, class_dist_dic, low_thresh, high_thresh):
    new_pred = pred.clone()
    new_prediction = prediction.clone()
    no_of_nodes = len(pred)
    no_of_classes = len(class_dist_dic)
    node_flag = {}
    for label_no in class_dist_dic:
        selected_nodes = np.where(pred==label_no)
        #print(selected_nodes[0], prediction[selected_nodes[0]])
        oneHotVec = np.zeros(no_of_nodes)
        oneHotVec[selected_nodes] = 1
        a = A.dot(oneHotVec)
        t = no_of_nodes//no_of_classes
        gate = (-np.sort(-a, axis=0))[t]
        index = np.where(a.flat > gate)[0]
        for node in index:
            if node not in node_flag:
                if max(prediction[node]) <= low_thresh:
                    flag = False
                    #if a nbd has higher conf than the source then label it with the nbds label
                    max_conf = max(prediction[selected_nodes[0][0]])
                    neighbors = [n for n in g.neighbors(node)]
                    neighbors = neighbors + [node]
                    for nbd in neighbors:
                        if max(prediction[nbd]) > max_conf:
                            new_pred[node] = new_pred[nbd]
                            max_conf = max(prediction[nbd])
                            flag = True
                    if flag == False:
                        new_pred[node] = label_no
                        node_flag[node] = True
    for node in range(no_of_nodes):
        new_prediction[node][new_pred[node].item()] = 1.0

    return new_prediction

def mean(data):
    n = len(data)
    mean = sum(data) / n
    return mean

def variance(data):
    n = len(data)
    mean = sum(data) / n
    deviations = [(x - mean) ** 2 for x in data]
    variance = sum(deviations) / n
    return variance

def stdev(data):
    import math
    var = variance(data)
    std_dev = math.sqrt(var)
    return std_dev

def getMeanAndStd(data):
    mu = mean(data)
    sigma = stdev(data)
    return mu, sigma


def olrw_parwalk(prediction, data, test_mask):
    adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=len(data.y))
    g = to_networkx(data, to_undirected=True, remove_self_loops=True)
    _,pred = prediction.max(dim=1)
    #extract the nodes of different labels: class distributions -- predicted labels
    class_dist_dic = collections.Counter(pred.numpy())
    #create the absorption probability
    A = absorption_probability(adj, alpha=1e-6)
    #get mu, sigma
    conf_list = [round(max(conf).item(),2) for conf in prediction]
    mu, sigma = getMeanAndStd(np.array(conf_list))

    max_acc = 0
    max_prediction = None

    for k in [0]:#np.arange(0, 2, 0.1):
        low_thresh = mu-k*sigma
        high_thresh = mu+k*sigma
        new_prediction = changePredParwalkHighNbdOrStartAllClassThresh(A, g, pred, prediction, class_dist_dic, low_thresh, high_thresh)
        _,new_pred = new_prediction.max(dim=1)
        _correct1 = new_pred[test_mask].eq(data.y[test_mask]).sum().item()
        acc = _correct1/len(data.y)
        if acc > max_acc:
            max_acc = acc
            max_prediction = new_prediction

    return max_prediction, max_acc

def changePredDeepwalkHighNbdOrStartAllClassThresh(model, g, pred, prediction, class_dist_dic, low_thresh, high_thresh):
    new_pred = pred.clone()
    new_prediction = prediction.clone()
    no_of_nodes = len(pred)
    no_of_classes = len(class_dist_dic)
    node_flag = {}
    model_config_t = no_of_nodes//no_of_classes
    for label_no in class_dist_dic:
        selected_nodes = np.where(pred==label_no)
        num_of_can = model_config_t//class_dist_dic[label_no]
        index = np.array([], dtype = int)
        for node in selected_nodes:
            if str(node) in model:
                similar_nodes = model.similar_by_word(str(node),topn=num_of_can)
                int_similar_nodes = np.array(list(set([int(tup[0]) for tup in similar_nodes])))
                index = np.concatenate((index, int_similar_nodes), axis=0)

        for node in index:
            if node not in node_flag:
                if max(prediction[node]) <= low_thresh:
                    flag = False
                    #if a nbd has higher conf than the source then label it with the nbds label
                    max_conf = max(prediction[selected_nodes[0][0]])
                    neighbors = [n for n in g.neighbors(node)]
                    neighbors = neighbors + [node]
                    for nbd in neighbors:
                        if max(prediction[nbd]) > max_conf:
                            new_pred[node] = new_pred[nbd]
                            max_conf = max(prediction[nbd])
                            flag = True
                    if flag == False:
                        new_pred[node] = label_no
                        node_flag[node] = True
    for node in range(no_of_nodes):
        new_prediction[node][new_pred[node].item()] = 1.0

    return new_prediction

def olrw_deepwalk(prediction, data, test_mask):
    adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=len(data.y))
    g = to_networkx(data, to_undirected=True, remove_self_loops=True)
    _,pred = prediction.max(dim=1)
    #extract the nodes of different labels: class distributions -- predicted labels
    class_dist_dic = collections.Counter(pred.numpy())
    deep_model = getDeepWalkModel(adj, class_dist_dic)
    #get mu, sigma
    conf_list = [round(max(conf).item(),2) for conf in prediction]
    mu, sigma = getMeanAndStd(np.array(conf_list))

    max_acc = 0
    max_prediction = None

    for k in np.arange(0, 2, 0.1):
        low_thresh = mu-k*sigma
        high_thresh = mu+k*sigma
        new_prediction = changePredDeepwalkHighNbdOrStartAllClassThresh(deep_model, g, pred, prediction, class_dist_dic, low_thresh, high_thresh)
        _,new_pred = new_prediction.max(dim=1)
        _correct1 = new_pred[test_mask].eq(data.y[test_mask]).sum().item()
        acc = _correct1/len(data.y)
        if acc > max_acc:
            max_acc = acc
            max_prediction = new_prediction

    return max_prediction, max_acc
