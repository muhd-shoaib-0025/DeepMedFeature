# DeepMedFeature
 Drug Drug Interactions (DDIs) are an important biological phenomena and can be a result of medical errors from medical practitioners. Drug interactions can change the molecular structure of interacting agents which may prove to be fatal in the worst case. Finding drug interactions early in diagnosis can play a key role in side-effect prevention. The growth of big data provides a rich source of information for clinical studies to investigate DDIs. We propose a hierarchical classification model which is double pass in nature. The first pass predicts the occurrence of an interaction and then the second pass further predicts the type of interaction such as effect, advice, mechanism, and int. We applied different deep learning algorithms with Convolutional Bi-LSTM (ConvBLSTM) proving to the best. The results show that pre-trained vector embeddings prove to be the most appropriate features. The F1-score of the ConvBLSTM algorithm turned out to be 96.39\% and 98.37\% in Russian and English language respectively which is greater than the state of the art systems. According to the results, it can be concluded that adding a convolution layer before the bi-directional pass improves model performance in the extraction and automatic classification of drug interactions, using pre-trained vector embeddings such as Fasttext and Bio-Bert. 
 
    python3 main.py -lang ru -sample_size 0.5 -num_hidden 512 -min_comp 256 -batch_size 64 -num_threads 32 -sampling over_sampling
