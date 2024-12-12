import pickle
import numpy as np
import argparse
import networkx as nx
from sklearn.metrics import roc_curve, auc
import pandas as pd
import obonet 
import multiprocessing as mp
from tqdm import tqdm 
import torch.nn as nn
import torch

# Placeholder for Go Graph; initialize with your data
graph = obonet.read_obo('PDB_test_set/go-basic.obo')

# Convert the graph to a directed graph
go_graph = nx.DiGraph()

# Add nodes with attributes
for node, data in graph.nodes(data=True):
    go_graph.add_node(node, **data)

# Add edges
for source, target in graph.edges():
    go_graph.add_edge(source, target)

def propagate_go_preds(Y_hat, goterms):
    go2id = {go: ii for ii, go in enumerate(goterms)}
    for goterm in goterms:
        if goterm in go_graph:
            parents = set(goterms).intersection(nx.descendants(go_graph,
                                                               goterm))
            for parent in parents:
                Y_hat[:, go2id[parent]] = np.maximum(Y_hat[:, go2id[goterm]],
                                                     Y_hat[:, go2id[parent]])

    return Y_hat

def evaluate_threshold(threshold, goterms, Y_true, Y_pred, prot2goterms, go_graph, ont2root, ont):
    n = Y_true.shape[0]
    predictions = (Y_pred > threshold).astype(int)
    precision, recall, m = 0.0, 0.0, 0

    pred_list = []
    true_list = []
    overlap_list = []
    for i in range(n):
        pred_gos = set()
        for goterm in goterms[np.where(predictions[i] == 1)[0]]:
            pred_gos = pred_gos.union(nx.descendants(go_graph, goterm))
            pred_gos.add(goterm)
        pred_gos.discard(ont2root[ont])

        num_pred = len(pred_gos)
        num_true = len(prot2goterms[i])
        num_overlap = len(prot2goterms[i].intersection(pred_gos))
        pred_list.append(num_pred)
        true_list.append(num_true)
        overlap_list.append(num_overlap)
        if num_pred > 0 and num_true > 0:
            m += 1
            precision += float(num_overlap) / num_pred
            recall += float(num_overlap) / num_true

    if m > 0:
        AvgPr = precision / m
        AvgRc = recall / n
        if AvgPr + AvgRc > 0:
            F_score = 2 * (AvgPr * AvgRc) / (AvgPr + AvgRc)
            return F_score, threshold, pred_list, true_list, overlap_list, Y_pred
    return 0, threshold, pred_list, true_list, overlap_list, Y_pred

def calculate_combined_f1(gobert_threshold, gobert_dict , baseline_threshold, baseline_dict, labels_dict, ont, sub_goterms):

    goterms = np.asarray(list(next(iter(labels_dict.values())).keys()))
    proteins = list(labels_dict.keys())
    prediction_dict = {}
    replaced_proteins = []
    for key, logits in labels_dict.items():
        count = 0
        for goterm, value in logits.items():
            if goterm in sub_goterms:
                count += value
        if count != 0:
            replaced_proteins.append(key)
    print(len(replaced_proteins))
    replaced_proteins1 = []
    for key, logits in baseline_dict.items():
        count = 0
        for goterm, value in logits.items():
            if goterm in sub_goterms and value > baseline_threshold:
                count+=1
        if count != 0:
            replaced_proteins1.append(key)
    replaced_proteins = list(set(replaced_proteins) & set(replaced_proteins1))        
    print(len(replaced_proteins), len(replaced_proteins1))

    for key, logits in baseline_dict.items():
        prediction_dict[key] = {}  # Initialize an empty dictionary for each key
        for goterm, value in logits.items():
            if key in replaced_proteins and goterm in gobert_dict[key]:
                if gobert_dict[key][goterm] > gobert_threshold:
                    prediction_dict[key][goterm] = 1
                else:
                    prediction_dict[key][goterm] = 0
            else:
                if value > baseline_threshold:
                    prediction_dict[key][goterm] = 1
                else:
                    prediction_dict[key][goterm] = 0

    Y_pred = np.array([[prediction_dict[p].get(go, 0) for go in goterms] for p in proteins])
    Y_true = np.array([[labels_dict[p].get(go, 0) for go in goterms] for p in proteins])
    n = Y_true.shape[0]
    predictions = Y_pred
    ont2root = {'bp': 'GO:0008150', 'mf': 'GO:0003674', 'cc': 'GO:0005575'}
    precision, recall, m = 0.0, 0.0, 0
    prot2goterms = {}
    for i in range(n):
        all_gos = set()
        for goterm in goterms[np.where(Y_true[i] == 1)[0]]:
            all_gos = all_gos.union(nx.descendants(go_graph, goterm))
            all_gos.add(goterm)
        all_gos.discard(ont2root[ont])
        prot2goterms[i] = all_gos

    pred_list = []
    true_list = []
    overlap_list  = []
    for i in range(n):
        pred_gos = set()
        for goterm in goterms[np.where(predictions[i] == 1)[0]]:
            pred_gos = pred_gos.union(nx.descendants(go_graph, goterm))
            pred_gos.add(goterm)
        pred_gos.discard(ont2root[ont])

        num_pred = len(pred_gos)
        num_true = len(prot2goterms[i])
        num_overlap = len(prot2goterms[i].intersection(pred_gos))
        pred_list.append(num_pred)
        true_list.append(num_true)
        overlap_list.append(num_overlap)
        if num_pred > 0 and num_true > 0:
            m += 1
            precision += float(num_overlap) / num_pred
            recall += float(num_overlap) / num_true


    if m > 0:
        AvgPr = precision / m
        AvgRc = recall / n
        if AvgPr + AvgRc > 0:
            F_score = 2 * (AvgPr * AvgRc) / (AvgPr + AvgRc)
            return F_score, pred_list, true_list, overlap_list
    return 0, pred_list, true_list, overlap_list

def evaluate_combine_threshold(gobert_threshold, gobert_ratio, gobert_dict, baseline_threshold, baseline_dict, labels_dict, ont, sub_goterms, goterms, proteins, ont2root):
    prediction_dict = {}
    replaced_proteins = []
    for key, logits in labels_dict.items():
        count = 0
        for goterm, value in logits.items():
            if goterm in sub_goterms:
                count += value
        if count != 0:
            replaced_proteins.append(key)

    replaced_proteins1 = []
    for key, logits in baseline_dict.items():
        count = 0
        for goterm, value in logits.items():
            if goterm in sub_goterms and value > baseline_threshold:
                count += 1
        if count != 0:
            replaced_proteins1.append(key)

    replaced_proteins = list(set(replaced_proteins) | set(replaced_proteins1))
    if gobert_threshold == 0.01 and gobert_ratio == 0.1:
        print(len(replaced_proteins))

    for key, logits in baseline_dict.items():
        prediction_dict[key] = {}
        for goterm, value in logits.items():
            if key in replaced_proteins and goterm in gobert_dict[key]:
                prediction_value = (gobert_dict[key][goterm]*gobert_ratio + value*(1-gobert_ratio))
                # if gobert_dict[key][goterm] > gobert_threshold:
                if prediction_value > gobert_threshold:
                    prediction_dict[key][goterm] = 1
                else:
                    prediction_dict[key][goterm] = 0
            else:
                if value > baseline_threshold:
                    prediction_dict[key][goterm] = 1
                else:
                    prediction_dict[key][goterm] = 0

    Y_pred = np.array([[prediction_dict[p].get(go, 0) for go in goterms] for p in proteins])
    Y_true = np.array([[labels_dict[p].get(go, 0) for go in goterms] for p in proteins])
    n = Y_true.shape[0]

    prot2goterms = {}
    for i in range(n):
        all_gos = set()
        for goterm in goterms[np.where(Y_true[i] == 1)[0]]:
            all_gos = all_gos.union(nx.descendants(go_graph, goterm))
            all_gos.add(goterm)
        all_gos.discard(ont2root[ont])
        prot2goterms[i] = all_gos

    precision, recall, m = 0.0, 0.0, 0
    pred_list = []
    true_list = []
    overlap_list = []
    for i in range(n):
        pred_gos = set()
        for goterm in goterms[np.where(Y_pred[i] == 1)[0]]:
            pred_gos = pred_gos.union(nx.descendants(go_graph, goterm))
            pred_gos.add(goterm)
        pred_gos.discard(ont2root[ont])

        num_pred = len(pred_gos)
        num_true = len(prot2goterms[i])
        num_overlap = len(prot2goterms[i].intersection(pred_gos))
        pred_list.append(num_pred)
        true_list.append(num_true)
        overlap_list.append(num_overlap)
        if num_pred > 0 and num_true > 0:
            m += 1
            precision += float(num_overlap) / num_pred
            recall += float(num_overlap) / num_true

    if m > 0:
        AvgPr = precision / m
        AvgRc = recall / n
        if AvgPr + AvgRc > 0:
            F_score = 2 * (AvgPr * AvgRc) / (AvgPr + AvgRc)
            return F_score, pred_list, true_list, overlap_list
    return 0, pred_list, true_list, overlap_list

def calculate_combined_f1_multiprocessing(gobert_thresholds, gobert_ratios, gobert_dict, baseline_threshold, baseline_dict, labels_dict, ont, sub_goterms):
    goterms = np.asarray(list(next(iter(labels_dict.values())).keys()))
    proteins = list(labels_dict.keys())
    ont2root = {'bp': 'GO:0008150', 'mf': 'GO:0003674', 'cc': 'GO:0005575'}

    # Prepare combinations of thresholds and ratios
    tasks = [
        (threshold, ratio, gobert_dict, baseline_threshold, baseline_dict, labels_dict, ont, sub_goterms, goterms, proteins, ont2root)
        for threshold in gobert_thresholds
        for ratio in gobert_ratios
    ]

    with mp.Pool(processes=int(mp.cpu_count()*0.8)) as pool:
        results = pool.starmap(evaluate_combine_threshold, tasks)

    # Find the best combination
    max_F_score = max(results)
    best_task_index = results.index(max_F_score)
    best_threshold, best_ratio = tasks[best_task_index][:2]

    return max_F_score, best_threshold, best_ratio

def calculate_f1_with_threshold(gobert_threshold, gobert_dict , baseline_threshold, baseline_dict, labels_dict, ont):
    goterms = list(next(iter(labels_dict.values())).keys())
    proteins = list(labels_dict.keys())
    # for key, logits in baseline_dict.items():
    #     for goterm, value in logits.items():
    #         if value > baseline_threshold:
    #             baseline_dict[key][goterm] = 1
    #         else:
    #             baseline_dict[key][goterm] = 0

    Y_pred = np.array([[baseline_dict[p].get(go, 0) for go in goterms] for p in proteins])
    Y_true = np.array([[labels_dict[p].get(go, 0) for go in goterms] for p in proteins])
    propagate_go_preds(Y_pred, goterms)

    n = Y_true.shape[0]
    # predictions = Y_pred
    goterms = np.asarray(goterms)
    ont2root = {'bp': 'GO:0008150', 'mf': 'GO:0003674', 'cc': 'GO:0005575'}
    prot2goterms = {}
    for i in range(n):
        all_gos = set()
        for goterm in goterms[np.where(Y_true[i] == 1)[0]]:
            all_gos = all_gos.union(nx.descendants(go_graph, goterm))
            all_gos.add(goterm)
        all_gos.discard(ont2root[ont])
        prot2goterms[i] = all_gos

    predictions = (Y_pred > baseline_threshold).astype(int)
    precision, recall, m = 0.0, 0.0, 0
    # print("\n",sum(Y_pred), baseline_threshold,"only baseline\n")
    pred_list = []
    true_list = []
    overlap_list  = []
    for i in range(n):
        pred_gos = set()
        for goterm in goterms[np.where(predictions[i] == 1)[0]]:
            pred_gos = pred_gos.union(nx.descendants(go_graph, goterm))
            pred_gos.add(goterm)
        pred_gos.discard(ont2root[ont])

        num_pred = len(pred_gos)
        num_true = len(prot2goterms[i])
        num_overlap = len(prot2goterms[i].intersection(pred_gos))
        pred_list.append(num_pred)
        true_list.append(num_true)
        overlap_list.append(num_overlap)
        if num_pred > 0 and num_true > 0:
            m += 1
            precision += float(num_overlap) / num_pred
            recall += float(num_overlap) / num_true


    if m > 0:
        AvgPr = precision / m
        AvgRc = recall / n
        if AvgPr + AvgRc > 0:
            F_score = 2 * (AvgPr * AvgRc) / (AvgPr + AvgRc)
            return F_score, pred_list, true_list, overlap_list
    return 0, pred_list, true_list, overlap_list

class Method:
    def __init__(self, model, subgraph, logits_dict, labels_dict, goterms=None, threshold=None):
        self.goterms = goterms or list(next(iter(labels_dict.values())).keys())
        self.proteins = list(labels_dict.keys())
        self.model = model
        if self.model == "GoBERT":
            sigmoid = nn.Sigmoid()  # Create an instance
            self.Y_pred = np.array(sigmoid(torch.tensor(np.array([[logits_dict[p].get(go, 0) for go in self.goterms] for p in self.proteins]))))
            # print(self.Y_pred, "\n GoBERT____________________")
        else:
            self.Y_pred = np.array([[logits_dict[p].get(go, 0) for go in self.goterms] for p in self.proteins])
        self.Y_true = np.array([[labels_dict[p].get(go, 0) for go in self.goterms] for p in self.proteins])
        self.ont = subgraph
        self.threshold = threshold

            
        self._propagate_preds()

    def _propagate_preds(self):
        self.Y_pred = propagate_go_preds(self.Y_pred, self.goterms)
    def fmax(self):
        labels = self.Y_true
        preds = self.Y_pred
        n = labels.shape[0]
        goterms = np.asarray(self.goterms)
        ont2root = {'bp': 'GO:0008150', 'mf': 'GO:0003674', 'cc': 'GO:0005575'}

        prot2goterms = {}
        for i in range(n):
            all_gos = set()
            for goterm in goterms[np.where(labels[i] == 1)[0]]:
                all_gos = all_gos.union(nx.descendants(go_graph, goterm))
                all_gos.add(goterm)
            all_gos.discard(ont2root[self.ont])
            prot2goterms[i] = all_gos

        if self.threshold:
            thresholds = [self.threshold]
        else:
            thresholds = [t / 100.0 for t in range(1, 100)]

        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.starmap(
                evaluate_threshold,
                [
                    (threshold, goterms, labels, preds, prot2goterms, go_graph, ont2root, self.ont)
                    for threshold in thresholds
                ]
            )

        max_F1, best_threshold, pred_list, true_list, overlap_list, predictions = max(results, key=lambda x: x[0])
        # print(f"pred_list{pred_list}")
        # print(f"true_list{true_list}")
        # print(f"overlap_list{overlap_list}")
        # print("\n",sum(predictions), thresholds)

        return max_F1, best_threshold

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Fmax for predictions")
    parser.add_argument('--ontology', choices=['BP', 'MF', 'CC'], required=True, help="Select ontology: BP, MF, or CC.")
    parser.add_argument('--model', choices=['DeepFri', 'PFresGO', 'HEAL', 'GoBERT'], required=True, help="Select model name.")
    parser.add_argument('--replace_gobert', action='store_true', help="Replace results with GoBERT predictions.")
    parser.add_argument('--test_subset', action='store_true', help="evaluate on small subset or not")
    parser.add_argument('--best_threshold', type=float, default=None,  help="best prediction threshold to save time")
    parser.add_argument('--best_gobert_threshold', type=float, default=None,  help="best prediction threshold for gobert to save time")
    parser.add_argument('--best_gobert_ratio', type=float, default=None, help="the best ratio to save time")
    args = parser.parse_args()

    # Determine file paths based on model and ontology
    ori_logits_dict = pickle.load(open(f"model_outputs/{args.model}/processed/{args.model}_{args.ontology}_logits.pkl", "rb"))
    labels_dict = pickle.load(open(f"PDB_test_set/{args.ontology}_labels.pkl", "rb"))
    subset_goterm=None
    if args.test_subset:
        subset_goterm = list(pd.read_csv(f"PDB_test_set/lowf1_{args.ontology}_goterms.txt", header=None)[0])
    method = Method(model = args.model,subgraph=args.ontology.lower(), logits_dict=ori_logits_dict, labels_dict=labels_dict, goterms=subset_goterm, threshold=args.best_threshold)

    if args.replace_gobert:
        subset_goterm =list(pd.read_csv(f"PDB_test_set/lowf1_{args.ontology}_goterms.txt", header=None)[0])
        gobert_dict = pickle.load(open(f"model_outputs/GoBERT/processed/GoBERT_{args.ontology}_logits.pkl", "rb"))
        if args.best_gobert_threshold:
            gobert_thresholds = [args.best_gobert_threshold]
        else:
            gobert_thresholds = [t / 100 for t in range(1, 100)]  # Thresholds from 0.01 to 0.99
        if args.best_gobert_ratio:
            gobert_ratios = [args.best_gobert_ratio]
        else:
            gobert_ratios = [t/10 for t in range(1,10)]
        max_F_score, best_threshold, best_ratio = calculate_combined_f1_multiprocessing(
            gobert_thresholds, 
            gobert_ratios,
            gobert_dict,
            baseline_threshold=args.best_threshold,
            baseline_dict=ori_logits_dict,
            labels_dict=labels_dict,
            ont=args.ontology.lower(),
            sub_goterms=subset_goterm  # Example sub_goterms
        )

        print(f"F-max for {args.model} with GoBERT on {args.ontology}: {max_F_score:.4f}, with GoBERT threshold {best_threshold} ratio {best_ratio}")
        gobert_method = Method(model = 'GoBERT',subgraph=args.ontology.lower(),logits_dict = gobert_dict, labels_dict=labels_dict, goterms=subset_goterm, threshold=best_threshold)
        gobert_fmax,gobert_threshold = gobert_method.fmax()
        print(f"{gobert_fmax:.4f}", gobert_threshold)

    else:
        # Calculate F-max
        fmax,thres = method.fmax()
        print(f"F-max for {args.model} on {args.ontology}: {fmax:.4f} with threshold {thres}")

