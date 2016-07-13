import numpy as np

#Extrapolates from a given seed sequence
def generate_from_seed(model, seed, sequence_length, data_variance, data_mean):
	seedSeq = seed.copy()
	output = []

	#The generation algorithm is simple:
	#Step 1 - Given A = [X_0, X_1, ... X_n], generate X_n + 1
	#Step 2 - Concatenate X_n + 1 onto A
	#Step 3 - Repeat MAX_SEQ_LEN times
	for it in xrange(sequence_length):
		seedSeqNew = np.array(model._predict(np.array([np.array([seedSeq[0][it:it+30]])]))) #Step 1. Generate X_n + 1
		print(seedSeq.shape, seedSeqNew.shape)
		#Step 2. Append it to the sequence
		if it == 0:
			for i in xrange(seedSeqNew.shape[2]):
				output.append(seedSeqNew[0][0][i].copy())
		else:
			output.append(seedSeqNew[0][0][-1].copy()) 
		newSeq = seedSeqNew[0][0][-1] # last stft predicted
		newSeq = np.reshape(newSeq, (1, 1, newSeq.shape[0]))
		print(newSeq.shape, seedSeq.shape)
		# seedSeq = np.concatenate((seedSeq, newSeq ), axis=1)

	#Finally, post-process the generated sequence so that we have valid frequencies
	#We're essentially just undo-ing the data centering process
	for i in xrange(len(output)):
		output[i] *= data_variance
		output[i] += data_mean
	return output
