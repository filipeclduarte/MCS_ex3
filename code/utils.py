###############################################################################
from functools import partial
from math import sqrt
from copy import deepcopy
import operator, sys

import json
import pickle
import pandas as pd
import numpy as np
from scipy.io import arff

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics.pairwise import euclidean_distances

from deslib.dcs import OLA, LCA, MCB
from deslib.des import KNORAE, KNORAU

# from two_stage_tiebreak_classifier import TSTBClassifier


def load_experiment_configuration():
	BAGGING_PERCENTAGE = 0.5
	N_JOBS = -1
	K_COMPETENCE = 7
	STRATEGY_PERCENTAGE = 0.5

	config = {
	"num_folds": 10,
	"pool_size": 100,
	# TODO: organizar para implementar o KDN para este experimento
	"kdn": 5,
	"strategy_percentage": STRATEGY_PERCENTAGE,
	"validation_hardnesses": _create_validation_hardnesses(threshold = 0.5),
	"k_competence": K_COMPETENCE,
	"base_classifier": partial(Perceptron, max_iter = 40, tol = 0.001,
		                       penalty = None, n_jobs = N_JOBS),
	# "base_classifier": partial(CalibratedClassifierCV, 
	# 							base_estimator=Perceptron(random_state=0), n_jobs=N_JOBS),
	"generation_strategy": partial(BaggingClassifier, 
		                           max_samples = BAGGING_PERCENTAGE,
		                           n_jobs = -1),
	"selection_strategies": _create_selection_strategies(K_COMPETENCE)
	}

	return config


def _create_validation_hardnesses(threshold):
	return [("None", partial(operator.gt, 2)), 
	        ("Hard", partial(operator.lt, threshold)), 
	        ("Easy", partial(operator.gt, threshold))]

def _find_k_neighbours(distances, k):
	
	matrix_neighbours = []
	for i in range(len(distances)):
		
		cur_neighbours = set()
		while len(cur_neighbours) < k:
			min_ix = np.argmin(distances[i])
			distances[i, min_ix] = sys.float_info.max

			if min_ix != i:
				cur_neighbours.add(min_ix)

		matrix_neighbours.append(list(cur_neighbours))

	return matrix_neighbours

def _calculate_kdn_hardness(instances, gold_labels, k):
	distances = euclidean_distances(instances, instances)
	neighbours = _find_k_neighbours(distances, k)

	hards = []
	for i in range(len(neighbours)):
		fixed_label = gold_labels[i]
		k_labels = gold_labels[neighbours[i]]
		dn = sum(map(lambda label: label != fixed_label, k_labels))
		hards.append(float(dn)/k)

	return hards

def select_validation_set(instances, labels, operator, k):
	hards = _calculate_kdn_hardness(instances, labels, k)
	filtered_triples = _filter_based_hardness(instances, labels, hards, operator)
	validation_instances = [t[0] for t in filtered_triples]
	validation_labels = [t[1] for t in filtered_triples]
	return np.array(validation_instances), validation_labels

def _filter_based_hardness(instances, labels, hards, op):
	triples = [(instances[i], labels[i], hards[i]) for i in range(len(hards))]
	res = list(filter(lambda t: op(t[2]), triples))
	return res


def _create_selection_strategies(k_competence):
	return [("F-KNU", "DES", partial(KNORAU, DFP=True, k=k_competence)),
	        # ("F-TS-KNU.0IH", "Hybrid", partial(KNORAU, DFP=True, k=k_competence,
	        # 	                            with_IH = True, IH_rate=0.01)),
	        ("F-KNE", "DES", partial(KNORAE, DFP=True, k=k_competence)),
	        # ("F-TS-KNE.0IH", "Hybrid", partial(KNORAE, DFP=True, k=k_competence,
	        #  	                            with_IH = True, IH_rate=0.01)),
	        ("F-OLA", "DCS", partial(OLA, DFP=True, k=k_competence)),
	        # ("F-TS-OLA.0IH", "Hybrid", partial(OLA, DFP=True, k=k_competence,
	        # 	                            with_IH = True, IH_rate=0.01)),
	        ("F-LCA", "DCS", partial(LCA, DFP=True, k=k_competence)),
			("F-MCB", "DCS", partial(MCB, DFP=True, k=k_competence)),
			("Bagging", "Estatica", partial(BaggingClassifier, DFP=True, k=k_competence))]
	        # ("F-TS-LCA.0IH", "Hybrid", partial(LCA, DFP=True, k=k_competence,
	        # 	                            with_IH = True, IH_rate=0.01)),
	        # ("F-TSTB.0IH-LCA", "Hybrid", partial(TSTBClassifier, 
	        # 	                                    selection_method='lca',
	        # 	                                    k=k_competence, DFP=True,
	        #                                         with_IH=True, IH_rate=0.01)),
	        # ("F-TSTB.0IH-OLA", "Hybrid", partial(TSTBClassifier,
	        # 	                                    selection_method='ola',
	        # 	                                    k=k_competence, DFP=True,
	        #                                         with_IH=True, IH_rate=0.01)),
	        # ("F-TSTB.2IH-LCA", "Hybrid", partial(TSTBClassifier,
	        # 	                                    selection_method='lca',
	        # 	                                    k=k_competence, DFP=True,
	        #                                         with_IH=True, IH_rate=0.20)),
	        # ("F-TSTB.2IH-OLA", "Hybrid", partial(TSTBClassifier,
	        # 	                                    selection_method='ola',
	        # 	                                    k=k_competence, DFP=True,
	        #                                         with_IH=True, IH_rate=0.20)),
	        # ("F-TSTB.4IH-LCA", "Hybrid", partial(TSTBClassifier,
	        # 	                                    selection_method='lca',
	        # 	                                    k=k_competence, DFP=True,
	        #                                         with_IH=True, IH_rate=0.40)),
	        # ("F-TSTB.4IH-OLA", "Hybrid", partial(TSTBClassifier,
	        # 	                                    selection_method='ola',
	        # 	                                    k=k_competence, DFP=True,
	        #                                         with_IH=True, IH_rate=0.40))]

def scale_data(train_instances, validation_instances, test_instances):
	scaler = StandardScaler()
	train_instances = scaler.fit_transform(train_instances)
	validation_instances = scaler.transform(validation_instances)
	test_instances = scaler.transform(test_instances)
	return train_instances, validation_instances, test_instances

def load_datasets_filenames():
	# filenames = ["cm1", "jm1"]
	filenames = ['cm1', 'pc1']
	return filenames

def load_dataset(set_filename):
	SET_PATH = "../data/"
	FILETYPE = ".arff"
	full_filepath = SET_PATH + set_filename + FILETYPE

	data, _ = arff.loadarff(full_filepath)

	dataframe = pd.DataFrame(data)
	dataframe.dropna(inplace=True)

	gold_labels = pd.DataFrame(dataframe["defects"])
	instances = dataframe.drop(columns = "defects")

	gold_labels = (gold_labels["defects"] == b'true').astype(int)

	return instances, gold_labels

def save_predictions(data):
	# with open('../predictions/all_predictions2.json', 'w') as outfile:
		# json.dump(data, outfile)

	with open('../predictions/all_predictions2.pkl', 'wb') as outfile:
		pickle.dump(data, outfile)

		
def load_predictions_data():
	# with open('../predictions/all_predictions2.json', 'r') as outfile:
		# return json.load(outfile)
	with open('../predictions/all_predictions2.pkl', 'rb') as outfile:
		return pickle.load(outfile)

def _error_score(gold_labels, predicted_labels):
	return 1 - accuracy_score(gold_labels, predicted_labels)

def _g1_score(gold_labels, predicted_labels, average):
	precision = precision_score(gold_labels, predicted_labels, average=average)
	recall = recall_score(gold_labels, predicted_labels, average=average)
	return sqrt(precision*recall)

# def _calculate_metrics(gold_labels, predicted_labels):
# def _calculate_metrics(gold_labels, data):
def _calculate_metrics(gold_labels, data):

	predicted_labels = data[0]
	# predicted_scores = data[2]
	
	metrics = {}
	# TODO: alterar o cálculo da curva roc para receber as probabilidades
	metrics["auc_roc"] = roc_auc_score(gold_labels, predicted_labels, average='macro')
	# metrics["auc_roc"] = roc_auc_score(gold_labels, predicted_scores, average='macro')
	metrics["g1"] = _g1_score(gold_labels, predicted_labels, average='macro')
	metrics["f1"] = f1_score(gold_labels, predicted_labels, average='macro')
	metrics["acc"] = accuracy_score(gold_labels, predicted_labels)

	return metrics

# def generate_metrics(predictions_dict):
# 	metrics = {}

# 	for set_name, set_dict in predictions_dict.items():
# 		metrics[set_name] = {}

# 		for fold, fold_dict in set_dict.items():

# 			gold_labels = fold_dict["gold_labels"]
# 			del fold_dict["gold_labels"]

# 			for strategy, data in fold_dict.items():

# 				fold_metrics = _calculate_metrics(gold_labels, data[0])

# 				if strategy not in metrics[set_name].keys():
# 				    metrics[set_name][strategy] = {"type": data[1], "metrics": [fold_metrics]}
# 				else:
# 					metrics[set_name][strategy]["metrics"].append(fold_metrics)

# 	return metrics

def _check_create_dict(given_dict, new_key):
	if new_key not in given_dict.keys():
		given_dict[new_key] = {}


# para incorporar hardness
def generate_metrics(predictions_dict):

	metrics = {}
	for set_name, set_dict in predictions_dict.items():
		metrics[set_name] = {}
		for fold, fold_dict in set_dict.items():        
			for hardness_type, filter_dict in fold_dict.items():
				_check_create_dict(metrics[set_name], hardness_type)
				gold_labels = fold_dict[hardness_type]["gold_labels"]
				del fold_dict[hardness_type]['gold_labels']
								
				for strategy, data_arr in filter_dict.items():
					metrics_str = metrics[set_name][hardness_type]
					# fold_metrics = _calculate_metrics(gold_labels, data_arr)
					fold_metrics = _calculate_metrics(gold_labels, data_arr)
					
					if strategy not in metrics_str.keys():
						metrics_str[strategy] = [fold_metrics]
					else:
						metrics_str[strategy].append(fold_metrics)

	return metrics

def _summarize_metrics_folds(metrics_folds):
	summary = {}
	metric_names = metrics_folds[0].keys()

	for metric_name in metric_names:
		scores = [metrics_folds[i][metric_name] for i in range(len(metrics_folds))]
		summary[metric_name] = [np.mean(scores), np.std(scores)]

	return summary

def summarize_metrics_folds(metrics_dict):

	summary = deepcopy(metrics_dict)

	for set_name, set_dict in metrics_dict.items():
		for hardness_type, data_dict in set_dict.items():
			for strategy_name, data_folds in data_dict.items():
				# cur_metrics_summary = _summarize_metrics_folds(data_folds['metrics'])
				cur_metrics_summary = _summarize_metrics_folds(data_folds)
				# summary[set_name][hardness_type][strategy_name] = {'metrics': cur_metrics_summary}, #'type': data_folds}
				summary[set_name][hardness_type][strategy_name] = cur_metrics_summary
		# for strategy_name, data_folds in set_dict.items():
			# cur_metrics_summary = _summarize_metrics_folds(data_folds["metrics"])
			# summary[set_name][strategy_name] = {"metrics": cur_metrics_summary,
												# "type": data_folds["type"]}
	with open('../salvando_resultados.pkl', 'wb') as outfile:
		pickle.dump(summary,outfile)

	return summary

def pandanize_summary(summary):

	# df = pd.DataFrame(columns = ['set', 'hardness','strategy', # type,
	#                   'mean_auc_roc', 'std_auc_roc', 'mean_acc', 'std_acc',
	#                   'mean_f1', 'std_f1', 'mean_g1', 'std_g1'])

	df = pd.DataFrame(columns = ['set', 'hardness', 'strategy',
	                  'mean_auc_roc', 'std_auc_roc', 'mean_acc', 'std_acc',
	                  'mean_f1', 'std_f1', 'mean_g1', 'std_g1'])


	# for set_name, set_dict in summary.items():
	# 	for strategy, summary_folds in set_dict.items():
	# 		df_folds = pd.DataFrame(_unfilled_row(3, 8),
	# 			                    columns = df.columns)
	# 		_fill_dataframe_folds(df_folds, summary_folds, set_name,
	# 			                  strategy)
	# 		df = df.append(df_folds)

	# return df.reset_index(drop = True)
	for set_name, set_dict in summary.items():
		for hardness_type, filter_dict in set_dict.items():
			for strategy, summary_folds in filter_dict.items():
				df_folds = pd.DataFrame(_unfilled_row(3, 8),
									columns = df.columns)
				_fill_dataframe_folds(df_folds, summary_folds, set_name,
									hardness_type, strategy)
				df = df.append(df_folds)

	return df.reset_index(drop = True)


def _unfilled_row(nb_str_columns, nb_float_columns):
	row = [" " for i in range(nb_str_columns)]
	row.extend([0.0 for j in range(nb_float_columns)])
	return [row]

def _fill_dataframe_folds(df, summary, set_name, hardness,strategy):
	df.at[0, "set"] = set_name
	df.at[0, "strategy"] = strategy
	df.at[0, 'hardness'] = hardness
	return _fill_dataframe_metrics(df, summary)

def _fill_dataframe_metrics(df, summary):
	for key, metrics in summary.items():
		df.at[0, "mean_" + key] = metrics[0]
		df.at[0, "std_" + key] = metrics[1]
	return df

def save_pandas_summary(df):
	pd.to_pickle(df, '../metrics/metrics_summary.pkl')

# def read_pandas_summary():
# 	return pd.read_pickle('../metrics/metrics_summary.pkl')

# def separate_pandas_summary(df, separate_sets):
# 	dfs = []

# 	if separate_sets == True:
# 		sets = df["set"].unique()
# 		for set_name in sets:
# 			dfs.append(df.loc[df["set"]==set_name])
# 	else:
# 		dfs.append(df)

# 	return dfs

# def write_comparison(dfs, focus_columns, filename):

# 	with open('../comparisons/'+ filename + '.txt', "w") as outfile:
# 		for df_set in dfs:
# 			if len(dfs) == 1:
# 				outfile.write("\n\nDATASET: Mixed\n")
# 			else:
# 				outfile.write("\n\nDATASET: " + df_set.iat[0,0] + "\n")
# 			outfile.write("Mean of metrics\n")
# 			outfile.write(df_set.groupby(by=focus_columns).mean().to_string())
# 			outfile.write("\n\nStd of metrics\n")
# 			outfile.write(df_set.groupby(by=focus_columns).std().fillna(0).to_string())
# 			outfile.write("\n")
# 			outfile.write("-------------------------------------------------")

def bool_str(s):

    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')

    return s == 'True'