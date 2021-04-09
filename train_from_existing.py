#! /usr/bin/env python3

import logging, sys
from simpletransformers.language_modeling import LanguageModelingModel, LanguageModelingArgs

def train(orig_bert, outdir_name, train_filename, eval_filename):
	logging.basicConfig(level=logging.INFO)
	transformers_logger = logging.getLogger("transformers")
	transformers_logger.setLevel(logging.WARNING)
	
	model_args = LanguageModelingArgs()
	model_args.reprocess_input_data = True
	model_args.output_dir = outdir_name
	model_args.best_model_dir = outdir_name + "/best_model"
	model_args.tensorboard_dir = outdir_name + "/runs"
	model_args.overwrite_output_dir = True
	model_args.save_steps = 0
	model_args.num_train_epochs = 10
	model_args.dataset_type = "simple"
	model_args.evaluate_during_training = True
	model_args.evaluate_during_training_verbose = True
	model_args.evaluate_during_training_steps = 0
	model_args.silent = True
	model_args.do_lower_case = ("uncased" in orig_bert)
	
	model = LanguageModelingModel("bert", orig_bert, args=model_args)
	model.train_model(train_filename, eval_file=eval_filename)

if __name__ == "__main__":
	casing = sys.argv[1]	# "cased" or "uncased"
	train("bert-base-german-dbmdz-{}".format(casing), "dbmdz-{}-swisscrawl".format(casing), "swisscrawl_reformatted.txt", "vardial_train_reformatted.txt")
