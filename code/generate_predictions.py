###############################################################################
import numpy as np
import random as rn

#DO NOT CHANGE THIS
np.random.seed(2577)
rn.seed(3581)
###################

from utils import load_datasets_filenames, load_experiment_configuration
from utils import load_dataset, save_predictions
from utils import scale_data
from utils import select_validation_set

from sklearn.model_selection import StratifiedKFold, train_test_split



if __name__ == "__main__":

	print("Step 1 - Loading configurations")

	datasets_filenames = load_datasets_filenames()
	config = load_experiment_configuration()
	predictions = {}
	exp = 1

	print("Step 2 - Starting experiment")

	for dataset_name in datasets_filenames:
		print('Dataset: ', dataset_name)
		instances, gold_labels = load_dataset(dataset_name)
		skfold = StratifiedKFold(n_splits = config["num_folds"],
			                     shuffle = True)

		predictions[dataset_name] = {}

		for fold, division in enumerate(skfold.split(X=instances, y=gold_labels), 1):
			# train_idxs = division[0] # adicionei agora
			test_idxs = division[1]
			# train_instances = instances.iloc[train_idxs].values # add agora
			# train_gold_labels = gold_labels.iloc[train_idxs].values.ravel() # add agora
			test_instances = instances.iloc[test_idxs].values
			test_gold_labels = gold_labels.iloc[test_idxs].values.ravel()

			predictions[dataset_name][fold] = {}
			# predictions[dataset_name][fold]["gold_labels"] = test_gold_labels.tolist()

			dev_idxs = division[0]
			dev_instances = instances.iloc[dev_idxs]
			dev_gold_labels = gold_labels.iloc[dev_idxs]

			# dev_tuple = train_test_split(dev_idxs, test_size = 0.2223, shuffle=True, 
			# 							stratify=dev_gold_labels)
			teste_gold_labels = True
			while teste_gold_labels:
				try:
					dev_tuple = train_test_split(dev_idxs, test_size = 0.25, shuffle=True, 
												stratify=dev_gold_labels)

					train_idxs = dev_tuple[0]
					train_instances = instances.iloc[train_idxs].values
					train_gold_labels = gold_labels.iloc[train_idxs].values.ravel()
					val_idxs = dev_tuple[1]
					validation_instances = instances.iloc[val_idxs].values
					validation_gold_labels = gold_labels.iloc[val_idxs].values.ravel()

					scaled_instances = scale_data(train_instances, validation_instances, test_instances)
					train_instances = scaled_instances[0]
					validation_instances = scaled_instances[1]
					test_instances = scaled_instances[2]


					#TODO: separacao dos dados em dificuldade das instancias com kDN
					for hardness_type, filter_func in config["validation_hardnesses"]:
						print('Hardness type: ', hardness_type)

						validation_instances, validation_gold_labels = select_validation_set(
							train_instances, train_gold_labels, filter_func, config["kdn"])
						
						predictions[dataset_name][fold][hardness_type] = {}

						# TODO: Testar se existe algum gold label com apenas uma classe
						
						teste_gold_labels = np.all(validation_gold_labels == validation_gold_labels[0])

						predictions[dataset_name][fold][hardness_type]['gold_labels'] = validation_gold_labels# add agora

						subpredictions = predictions[dataset_name][fold][hardness_type]

						base_clf = config["base_classifier"]()
						clf_pool = config["generation_strategy"](base_clf, config["pool_size"])
						clf_pool.fit(train_instances, train_gold_labels)
					
					# gerando o pool de classificadores com o bagging de 100 perceptrons
					# base_clf = config["base_classifier"]()
					# clf_pool = config["generation_strategy"](base_clf, config["pool_size"])
					# clf_pool.fit(train_instances, train_gold_labels)

						# TODO: testar com bagging 
						for strategy_name, strategy_type, selection_func in config["selection_strategies"]:
							
							print('Strategy: ', strategy_name)
							if strategy_name != 'Bagging':

								ds_clf = selection_func(pool_classifiers=clf_pool)
								ds_clf.fit(validation_instances, validation_gold_labels)
								# ds_clf.fit(train_instances, train_gold_labels)
								# print(ds_clf.predict(test_instances).astype(int))
								# cur_predictions = ds_clf.predict(test_instances).astype(int)
								cur_predictions = ds_clf.predict(validation_instances).astype(int) # add agora
								# TODO: Adicionar a previsão de probabilidade para calcular a curva ROC
								# cur_predictions_proba = ds_clf.predict_proba(validation_instances)[:,1]
								data_arr = [cur_predictions.tolist(), strategy_type]
								# data_arr = [cur_predictions.tolist(), strategy_type, cur_predictions_proba.tolist()]
								# predictions[dataset_name][fold][strategy_name] = [cur_predictions.tolist(), strategy_type]
								subpredictions[strategy_name] = data_arr
							else:

								st_clf = clf_pool
								# cur_predictions = st_clf.predict(test_instances).astype(int)
								cur_predictions = st_clf.predict(validation_instances).astype(int) # add agora
								# TODO: Adicionar a previsão de probabilidade para calcular a curva ROC
								# cur_predictions_proba = st_clf.predict_proba(validation_instances)[:,1]
								# predictions[dataset_name][fold][strategy_name] = [cur_predictions.tolist(), strategy_type]
								data_arr = [cur_predictions.tolist(), strategy_type]
								# data_arr = [cur_predictions.tolist(), strategy_type, cur_predictions_proba.tolist()]
								subpredictions[strategy_name] = data_arr

							

							print("Experiment " + str(exp))
							exp+=1
				
				except:
					pass


			

				

	print("Step 2 - Finished experiment")

	print("Step 3 - Storing predictions")
	save_predictions(predictions)