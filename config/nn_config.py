def get_neural_net_configuration():
	model_id = 'piano-vibra'
	
	nn_params = {}
	nn_params['sampling_frequency'] = 44100
	#Number of hidden dimensions.
	#For best results, this should be >= freq_space_dims, but most consumer GPUs can't handle large sizes
	nn_params['hidden_dimension_size'] = 1024
	#The weights filename for saving/loading trained models
	nn_params['model_basename'] = './'+model_id
	#The model filename for the training data
	nn_params['model_file'] = './datasets/'+model_id
	#The dataset directory
	nn_params['dataset_directory'] = './datasets/'+model_id+'/'
	return nn_params
