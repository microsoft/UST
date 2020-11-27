"""
Author: Subhabrata Mukherjee (submukhe@microsoft.com)
Code for Uncertainty-aware Self-training (UST) for few-shot learning.
"""

from huggingface_utils import MODELS
from preprocessing import generate_sequence_data
from sklearn.utils import shuffle
from transformers import *
from ust import train_model

import argparse
import logging
import numpy as np
import os
import random
import sys

# logging
logger = logging.getLogger('UST')
logging.basicConfig(level = logging.INFO)

GLOBAL_SEED = int(os.getenv("PYTHONHASHSEED"))
logger.info ("Global seed {}".format(GLOBAL_SEED))

if __name__ == '__main__':

	# construct the argument parse and parse the arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--task", required=True, help="path of the task directory containing train, test and unlabeled data files")
	parser.add_argument("--model_dir", required=True, help="path to store model files")
	parser.add_argument("--seq_len", required=True, type=int, help="sequence length")
	parser.add_argument("--sup_batch_size", nargs="?", type=int, default=4, help="batch size for fine-tuning base model")
	parser.add_argument("--unsup_batch_size", nargs="?", type=int, default=32, help="batch size for self-training on pseudo-labeled data")
	parser.add_argument("--sample_size", nargs="?", type=int, default=16384, help="number of unlabeled samples for evaluating uncetainty on in each self-training iteration")
	parser.add_argument("--unsup_size", nargs="?", type=int, default=4096, help="number of pseudo-labeled instances drawn from sample_size and used in each self-training iteration")
	parser.add_argument("--sample_scheme", nargs="?", default="easy_bald_class_conf", help="Sampling scheme to use")
	parser.add_argument("--sup_labels", nargs="?", type=int, default=60, help="number of labeled samples per class for training and validation (total)")
	parser.add_argument("--T", nargs="?", type=int, default=30, help="number of masked models for uncertainty estimation")
	parser.add_argument("--alpha", nargs="?", type=float, default=0.1, help="hyper-parameter for confident training loss")
	parser.add_argument("--valid_split", nargs="?", type=float, default=0.5, help="percentage of sup_labels to use for validation for each class")
	parser.add_argument("--sup_epochs", nargs="?", type=int, default=70, help="number of epochs for fine-tuning base model")
	parser.add_argument("--unsup_epochs", nargs="?", type=int, default=25, help="number of self-training iterations")
	parser.add_argument("--N_base", nargs="?", type=int, default=5, help="number of times to randomly initialize and fine-tune few-shot base encoder to select the best starting configuration")
	parser.add_argument("--pt_teacher", nargs="?", default="TFBertModel",help="Pre-trained teacher model")
	parser.add_argument("--pt_teacher_checkpoint", nargs="?", default="bert-base-uncased", help="teacher model checkpoint to load pre-trained weights")
	parser.add_argument("--do_pairwise", action="store_true", default=False, help="whether to perform pairwise classification tasks like MNLI")
	parser.add_argument("--hidden_dropout_prob", nargs="?", type=float, default=0.2, help="dropout probability for hidden layer of teacher model")
	parser.add_argument("--attention_probs_dropout_prob", nargs="?", type=float, default=0.2, help="dropout probability for attention layer of teacher model")
	parser.add_argument("--dense_dropout", nargs="?", type=float, default=0.5, help="dropout probability for final layers of teacher model")

	args = vars(parser.parse_args())
	logger.info(args)

	task_name = args["task"]
	max_seq_length = args["seq_len"]
	sup_batch_size = args["sup_batch_size"]
	unsup_batch_size = args["unsup_batch_size"]
	unsup_size = args["unsup_size"]
	sample_size = args["sample_size"]
	model_dir = args["model_dir"]
	sample_scheme = args["sample_scheme"]
	sup_labels = args["sup_labels"]
	T = args["T"]
	alpha = args["alpha"]
	valid_split = args["valid_split"]
	sup_epochs = args["sup_epochs"]
	unsup_epochs = args["unsup_epochs"]
	N_base = args["N_base"]
	pt_teacher = args["pt_teacher"]
	pt_teacher_checkpoint = args["pt_teacher_checkpoint"]
	do_pairwise = args["do_pairwise"]
	dense_dropout = args["dense_dropout"]
	attention_probs_dropout_prob = args["attention_probs_dropout_prob"]
	hidden_dropout_prob = args["hidden_dropout_prob"]

	#Get pre-trained model, tokenizer and config
	for indx, model in enumerate(MODELS):
		if model[0].__name__ == pt_teacher:
			TFModel, Tokenizer, Config = MODELS[indx]

	#get pre-trained tokenizer
	tokenizer = Tokenizer.from_pretrained(pt_teacher_checkpoint)

	X_train_all, y_train_all = generate_sequence_data(max_seq_length, task_name+ "/train.tsv" ,tokenizer, do_pairwise=do_pairwise)

	X_test, y_test = generate_sequence_data(max_seq_length, task_name + "/test.tsv", tokenizer, do_pairwise=do_pairwise)

	X_unlabeled, _ = generate_sequence_data(max_seq_length, task_name + "/transfer.txt", tokenizer, unlabeled=True, do_pairwise=do_pairwise)


	for i in range(3):
		logger.info("***Train***")
		logger.info ("Example {}".format(i))
		logger.info ("Token ids {}".format(X_train_all["input_ids"][i]))
		logger.info (tokenizer.convert_ids_to_tokens(X_train_all["input_ids"][i]))
		#logger.info ("Token type ids {}".format(X_train_all["token_type_ids"][i]))
		logger.info ("Token mask {}".format(X_train_all["attention_mask"][i]))
		logger.info ("Label {}".format(y_train_all[i]))

	for i in range(3):
		logger.info("***Test***")
		logger.info ("Example {}".format(i))
		logger.info ("Token ids {}".format(X_test["input_ids"][i]))
		logger.info (tokenizer.convert_ids_to_tokens(X_test["input_ids"][i]))
		#logger.info ("Token type ids {}".format(X_test["token_type_ids"][i]))
		logger.info ("Token mask {}".format(X_test["attention_mask"][i]))
		logger.info ("Label {}".format(y_test[i]))

	for i in range(3):
		logger.info("***Unlabeled***")
		logger.info ("Example {}".format(i))
		logger.info ("Token ids {}".format(X_unlabeled["input_ids"][i]))
		logger.info (tokenizer.convert_ids_to_tokens(X_unlabeled["input_ids"][i]))
		#logger.info ("Token type ids {}".format(X_unlabeled["token_type_ids"][i]))
		logger.info ("Token mask {}".format(X_unlabeled["attention_mask"][i]))

	#labels indexed from 0
	labels = set(y_train_all)
	if 0 not in labels:
		y_train_all -= 1
		y_test -= 1 	
	labels = set(y_train_all)	
	logger.info ("Labels {}".format(labels))

	#if sup_labels < 0, then use all training labels in train file for learning
	if sup_labels < 0:
		X_train = X_train_all
		y_train = y_train_all
	else:
		X_input_ids, X_token_type_ids, X_attention_mask, y_train = [], [], [], []
		for i in labels:
			#get sup_labels from each class
			indx = np.where(y_train_all==i)[0]
			random.Random(GLOBAL_SEED).shuffle(indx)
			indx = indx[:sup_labels]
			X_input_ids.extend(X_train_all["input_ids"][indx])
			X_token_type_ids.extend(X_train_all["token_type_ids"][indx])
			X_attention_mask.extend(X_train_all["attention_mask"][indx])
			y_train.extend(np.full(sup_labels, i))

		X_input_ids, X_token_type_ids, X_attention_mask, y_train = shuffle(X_input_ids, X_token_type_ids, X_attention_mask, y_train, random_state=GLOBAL_SEED)

		X_train = {"input_ids": np.array(X_input_ids), "token_type_ids": np.array(X_token_type_ids), "attention_mask": np.array(X_attention_mask)}
		y_train = np.array(y_train)

	train_model(max_seq_length, X_train, y_train, X_test, y_test, X_unlabeled, model_dir, tokenizer, sup_batch_size=sup_batch_size, unsup_batch_size=unsup_batch_size, unsup_size=unsup_size, sample_size=sample_size, TFModel=TFModel, Config=Config, pt_teacher_checkpoint=pt_teacher_checkpoint, sample_scheme=sample_scheme, T=T, alpha=alpha, valid_split=valid_split, sup_epochs=sup_epochs, unsup_epochs=unsup_epochs, N_base=N_base, dense_dropout=dense_dropout, attention_probs_dropout_prob=attention_probs_dropout_prob, hidden_dropout_prob=hidden_dropout_prob)

	# train_model(max_seq_length, tokenizer, sup_batch_size, unsup_size, sample_size, X_train, y_train, X_test, y_test, X_unlabeled, model_dir, TFModel, Config, pt_teacher_checkpoint, sample_scheme, T, alpha, valid_split, sup_epochs, unsup_epochs, N_base)
