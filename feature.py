import concurrent
from abc import abstractmethod
from collections import Counter
from datetime import datetime
import fasttext.util
import fasttext
import pytz
from nltk.util import ngrams
from itertools import combinations
import numpy as np
from torch.autograd import Variable
import math
import copy
import nltk
from nltk import PerceptronTagger
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import multiprocessing as mp
from transformers import AutoModel, AutoTokenizer, BertTokenizer
from bisect import bisect_left
import threading

class Feature:
    def __init__(self, name, k, type, num_threads):
        self.name = name
        self.k = k
        self.feature_df=None
        self.feature_size=None
        self.type=type
        self.num_threads = num_threads

    @abstractmethod
    def create_df(self, df):
        pass

    @abstractmethod
    def compute_feature_matrix(self, df, X, feature_matrix):
        pass

    def set_feature_size(self, feature_size):
        self.feature_size = feature_size

class FrequentWords(Feature):

    def __init__(self, name, k, type, num_threads):
        super().__init__(name, k, type, num_threads)

    def compute_frequent_words(self, df, word_counter):
        for _, row in tqdm(df.iterrows(), desc='Computing frequent_words ... #rows=' + str(len(df)) +
                                               '; time: ' + datetime.now(pytz.timezone("Asia/Karachi")).strftime("%I:%M:%S%p") +
                                               '; cpu: ' + mp.current_process().name +
                                               '; thread: ' + str(threading.get_ident())):
            unique_tokens = set(word for word in row.normalized_sentence.split())
            bi_grams = ngrams(row.normalized_sentence.split(), 2)
            word_counter[0] += Counter([' '.join(bi_gram).strip() for bi_gram in bi_grams])
            word_counter[0] += Counter(unique_tokens)

    def compute_feature_matrix(self, df, X, feature_matrix):
        for _, row in tqdm(df.iterrows(), desc='Computing word feature_matrix ... #rows=' + str(len(df)) +
                                               '; time: ' + datetime.now(pytz.timezone("Asia/Karachi")).strftime("%I:%M:%S%p") +
                                               '; cpu: ' + mp.current_process().name +
                                               '; thread: ' + str(threading.get_ident())):
            feature_vector = [0 for i in range(0, len(X))]
            tokens = row.normalized_sentence.split()
            bi_grams = ngrams(tokens, 2)
            for word in tokens:
                if word in X:
                    index = X.index(word)
                    feature_vector[index] = feature_vector[index] + 1
            for bi_gram in bi_grams:
                if bi_gram in X:
                    index = X.index(bi_gram)
                    feature_vector[index] = feature_vector[index] + 1
            feature_matrix[_] = feature_vector

    def create_df(self, df):
        dfs = np.array_split(df, self.num_threads)
        word_counter = mp.Manager().list([Counter()])
        threads = list()
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            for df in dfs:
                t = executor.submit(self.compute_frequent_words, df, word_counter)
                threads.append(t)
            executor.shutdown(wait=True)
        frequent_words = sorted(list(dict(word_counter[0].most_common(self.k)).keys()))
        feature_matrix = mp.Manager().dict()
        threads = list()
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            for df in dfs:
                t = executor.submit(self.compute_feature_matrix, df, frequent_words, feature_matrix)
                threads.append(t)
            executor.shutdown(wait=True)
        return feature_matrix

class FrequentPhrases(Feature):

    def __init__(self, name, k, type, num_threads):
        super().__init__(name, k, type, num_threads)

    def compute_frequent_phrases(self, df, pair_counter):
        for _, row in tqdm(df.iterrows(), desc='Computing frequent_phrase... #rows=' + str(len(df)) +
                                               '; time: ' + datetime.now(pytz.timezone("Asia/Karachi")).strftime("%I:%M:%S%p") +
                                               '; cpu: ' + mp.current_process().name +
                                               '; thread: ' + str(threading.get_ident())):
            unique_tokens = set(word for word in row.normalized_sentence.split())
            combos = list(combinations(unique_tokens, 2))
            pair_counter[0] += Counter(combos)


    def BinarySearch(self, a, x):
        i = bisect_left(a, x)
        if i != len(a) and a[i] == x:
            return i
        else:
            return -1

    def compute_feature_matrix(self, df, X, feature_matrix):
        for _, row in tqdm(df.iterrows(), desc='Computing phrases feature_matrix... #rows=' + str(len(df)) +
                                               '; time: ' + datetime.now(pytz.timezone("Asia/Karachi")).strftime("%I:%M:%S%p") +
                                               '; cpu: ' + mp.current_process().name +
                                               '; thread: ' + str(threading.get_ident())):
            feature_vector = [0 for i in range(0, len(X))]
            unique_tokens = set(word for word in row.normalized_sentence.split())
            combos = list(combinations(unique_tokens, 2))
            for word in combos:
                index = self.BinarySearch(X, word)
                if index != -1:
                    feature_vector[index] = feature_vector[index] + 1
            feature_matrix[_] = feature_vector

    def create_df(self, df):
        dfs = np.array_split(df, self.num_threads)
        pair_counter = mp.Manager().list([Counter()])
        threads = list()
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            for df in dfs:
                t = executor.submit(self.compute_frequent_phrases, df, pair_counter)
                threads.append(t)
            executor.shutdown(wait=True)
        frequent_phrases = sorted(list(dict(pair_counter[0].most_common(self.k)).keys()))
        feature_matrix = mp.Manager().dict()
        threads = list()
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            for df in dfs:
                t = executor.submit(self.compute_feature_matrix, df, frequent_phrases, feature_matrix)
                threads.append(t)
            executor.shutdown(wait=True)
        return feature_matrix

class SyntacticGrammarTrios(Feature):

    def __init__(self, name, k, type, num_threads):
        super().__init__(name, k, type, num_threads)

    def compute_frequent_pos(self, df, chunker, trio_counter):
        for _, row in tqdm(df.iterrows(), desc='Computing frequent pos... #rows=' + str(len(df)) +
                                               '; time: ' + datetime.now(pytz.timezone("Asia/Karachi")).strftime("%I:%M:%S%p") +
                                               '; cpu: ' + mp.current_process().name +
                                               '; thread: ' + str(threading.get_ident())):
            chunk_dict = chunker[0].chunk_sentence(row.normalized_sentence)
            trigrams_list = []
            for key, pos_tagged_sentences in chunk_dict.items():
                pos_tags = [token[1] for pos_tagged_sentence in pos_tagged_sentences for token in pos_tagged_sentence]
                if len(pos_tags) > 2:
                    trigrams = ngrams(pos_tags, 3)
                    trigrams_list = [' '.join(trigram) for trigram in trigrams]
            trio_counter[0] += Counter(trigrams_list)


    def compute_feature_matrix(self, df, X, feature_matrix):

        for _, row in tqdm(df.iterrows(), desc='Computing pos feature_matrix... #rows=' + str(len(df)) +
                                               '; time: ' + datetime.now(pytz.timezone("Asia/Karachi")).strftime("%I:%M:%S%p") +
                                               '; cpu: ' + mp.current_process().name +
                                               '; thread: ' + str(threading.get_ident())):
            feature_vector = [0 for i in range(0, len(X[0]))]
            chunk_dict = X[1].chunk_sentence(row.normalized_sentence)
            trigrams_list = []
            for key, pos_tagged_sentences in chunk_dict.items():
                pos_tags = [token[1] for pos_tagged_sentence in pos_tagged_sentences for token in pos_tagged_sentence]
                if len(pos_tags) > 2:
                    trigrams = ngrams(pos_tags, 3)
                    trigrams_list = [' '.join(trigram) for trigram in trigrams]
            for word in trigrams_list:
                if word in X[0]:
                    index = X[0].index(word)
                    feature_vector[index] = feature_vector[index] + 1
            feature_matrix[_] = feature_vector

    def create_df(self, df):
        dfs = np.array_split(df, self.num_threads)
        trio_counter = mp.Manager().list([Counter()])
        grammar = self.PatternGrammar().compile_syntactic_grammar(0)
        chunker = self.Chunker(grammar)
        threads = list()
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            for df in dfs:
                t = executor.submit(self.compute_frequent_pos, df, mp.Manager().list([chunker]), trio_counter)
                threads.append(t)
            executor.shutdown(wait=True)

        trio_counter = sorted(list(dict(trio_counter[0].most_common(self.k)).keys()))
        feature_matrix = mp.Manager().dict()
        threads = list()
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            for df in dfs:
                t = executor.submit(self.compute_feature_matrix, df, [trio_counter, chunker], feature_matrix)
                threads.append(t)
            executor.shutdown(wait=True)

        return feature_matrix

    class PatternGrammar:
        @property
        def syntactic_grammars(self):
            grammar = {
                0: """
                    JJ_VBG_RB_DESCRIBING_NN: {   (<CC|,>?<JJ|JJ.>*<VB.|V.>?<NN|NN.>)+<RB|RB.>*<MD>?<WDT|DT>?<VB|VB.>?<RB|RB.>*(<CC|,>?<RB|RB.>?<VB|VB.|JJ.|JJ|RB|RB.>+)+}
                    """,
                1: """
                        VBG_DESRIBING_NN: {<NN|NN.><VB|VB.>+<RB|RB.>*<VB|VB.>}
                    """,
            }
            return grammar

        def compile_syntactic_grammar(self, index):
            return nltk.RegexpParser(self.syntactic_grammars[index])

    class PosTagger:
        def __init__(self, sentence):
            """

            Args:
                sentence:
            """
            self.sentence = sentence
            self.tagger = self.get_tagger()

        def pos_tag(self):
            """

            Returns:

            """
            tokens = nltk.word_tokenize(self.sentence)
            pos_tagged_tokens = self.tagger.tag(tokens)
            return pos_tagged_tokens

        @staticmethod
        def get_tagger():
            """

            Returns:

            """
            return PerceptronTagger()

    class Chunker:
        def __init__(self, grammar: nltk.RegexpParser):
            self.grammar = grammar

        def chunk_sentence(self, sentence: str):
            pos_tagged_sentence = SyntacticGrammarTrios.PosTagger(sentence).pos_tag()
            return dict(self.chunk_pos_tagged_sentence(pos_tagged_sentence))

        def chunk_pos_tagged_sentence(self, pos_tagged_sentence):
            chunked_tree = self.grammar.parse(pos_tagged_sentence)
            chunk_dict = self.extract_rule_and_chunk(chunked_tree)
            return chunk_dict

        def extract_rule_and_chunk(self, chunked_tree: nltk.Tree) -> dict:
            def recursively_get_pos_only(tree, collector_list=None, depth_limit=100):
                if collector_list is None:
                    collector_list = []
                if depth_limit <= 0:
                    return collector_list
                for subtree in tree:
                    if isinstance(subtree, nltk.Tree):
                        recursively_get_pos_only(subtree, collector_list, depth_limit - 1)
                    else:
                        collector_list.append(subtree)
                return collector_list

            def get_pos_tagged_and_append_to_chunk_dict(chunk_dict, subtrees):  # params can be removed now
                pos_tagged = recursively_get_pos_only(subtrees)
                chunk_dict[subtrees.label()].append(pos_tagged)

            chunk_dict = nltk.defaultdict(list)
            for subtrees in chunked_tree:
                if isinstance(subtrees, nltk.Tree):
                    get_pos_tagged_and_append_to_chunk_dict(chunk_dict, subtrees)
                    for sub in subtrees:
                        if isinstance(sub, nltk.Tree):
                            get_pos_tagged_and_append_to_chunk_dict(chunk_dict, sub)
            return chunk_dict

class RDF2Vec(Feature):

    def __init__(self, name, k, type, num_threads):
        super().__init__(name, k, type, num_threads)

    def compute_feature_matrix(self, df, X, feature_matrix):
        for _, row in tqdm(df.iterrows(), desc='Computing rdf2vec feature_matrix... #rows=' + str(len(df)) +
                                               '; time: ' + datetime.now(pytz.timezone("Asia/Karachi")).strftime("%I:%M:%S%p") +
                                               '; cpu: ' + mp.current_process().name +
                                               '; thread: ' + str(threading.get_ident())):
            try:
                drug1 = row['drug1'].lower()
                drug2 = row['drug2'].lower()

                feature_vector1 = []
                feature_vector2 = []
                if drug1 in X:
                    feature_vector1 = X[drug1]
                elif len(feature_matrix) > 0:
                    feature_vector1 = np.mean([np.array(feature_matrix[key]) for key in feature_matrix], axis=0)

                if drug2 in X:
                    feature_vector2 = X[drug2]
                elif len(feature_matrix) > 0:
                    feature_vector2 = np.mean([np.array(feature_matrix[key]) for key in feature_matrix], axis=0)

                if feature_vector1 and feature_vector2:
                    feature_vector = feature_vector1 + feature_vector2
                    feature_matrix[_] = feature_vector
            except:
                continue

    def create_df(self, df):
        features = dict()
        file = open('rdf2vec.txt', 'r')
        file.readline()
        lines = file.readlines()
        file.close()
        for line in lines:
            drug = line.split('\t')[0]
            feature = line.split('\t')[1].split(' ')
            feature = [float(i) for i in feature]
            features[drug] = feature
        dfs = np.array_split(df, self.num_threads)
        feature_matrix = mp.Manager().dict()
        threads = list()
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            for df in dfs:
                t = executor.submit(self.compute_feature_matrix, df, features, feature_matrix)
                threads.append(t)
            executor.shutdown(wait=True)

        return feature_matrix


class BioBert_ru(Feature):

    def __init__(self, name, k, type, num_threads):
        super().__init__(name, k, type, num_threads)

    def compute_feature_matrix(self, df, X, feature_matrix):

        with torch.no_grad():
            for _, row in tqdm(df.iterrows(), desc='Computing bert feature_matrix... #rows=' + str(len(df)) +
                                                   '; time: ' + datetime.now(pytz.timezone("Asia/Karachi")).strftime(
                "%I:%M:%S%p") +
                                                   '; cpu: ' + mp.current_process().name +
                                                   '; thread: ' + str(threading.get_ident())):
                try:
                    inputs = X[1](row.normalized_sentence, max_length=768, padding=True, truncation=True, return_tensors="pt")
                    outputs = X[0](**inputs)
                    embeddings = outputs.last_hidden_state
                    feature_vector = embeddings.reshape([embeddings.shape[1], embeddings.shape[2]]).tolist()
                    feature_vector = list(map(lambda x: sum(x) / len(x), zip(*feature_vector)))
                    feature_matrix[_] = feature_vector
                except:
                    continue

    def create_df(self, df):
        model = AutoModel.from_pretrained("cimm-kzn/rudr-bert")
        tokenizer = BertTokenizer.from_pretrained("cimm-kzn/rudr-bert")
        dfs = np.array_split(df, self.num_threads)
        feature_matrix = mp.Manager().dict()
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            for df in dfs:
                executor.submit(self.compute_feature_matrix, df, [model, tokenizer], feature_matrix)
            executor.shutdown(wait=True)
        return feature_matrix

'''
class BioBert_en(Feature):

    def __init__(self, name, k, type, num_threads):
        super().__init__(name, k, type, num_threads)

    def bert_text_preparation(self, text, tokenizer):
        marked_text = "[CLS] " + text + " [SEP]"
        tokenized_text = tokenizer.tokenize(marked_text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(indexed_tokens)
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        return tokenized_text, tokens_tensor, segments_tensors

    def get_bert_embeddings(self, tokens_tensor, segments_tensors, model):
        with torch.no_grad():
            outputs = model(tokens_tensor, segments_tensors)
            hidden_states = outputs.last_hidden_state
            hidden_states = hidden_states.reshape([hidden_states.shape[1], hidden_states.shape[2]]).tolist()
        return hidden_states

    def compute_feature_matrix(self, df, X, feature_matrix):
        for _, row in tqdm(df.iterrows(), desc='Computing biobert feature_matrix... #rows=' + str(len(df)) +
                                              '; time: ' + datetime.now(pytz.timezone("Asia/Karachi")).strftime("%I:%M:%S%p") +
                                               '; cpu: ' +  mp.current_process().name +
                                               '; thread: ' + str(threading.get_ident())):
            tokenized_text, tokens_tensor, segments_tensors = self.bert_text_preparation(row.normalized_sentence, X[1])
            feature_vector = self.get_bert_embeddings(tokens_tensor, segments_tensors, X[0])
            feature_vector = list(map(lambda x: sum(x) / len(x), zip(*feature_vector)))
            feature_matrix[_] = feature_vector

    def create_df(self, df):
        model_name = "biobert_en"
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        dfs = np.array_split(df, self.num_threads)
        feature_matrix = mp.Manager().dict()
        X = mp.Manager().list([model, tokenizer])
        threads = list()
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            for df in dfs:
                t = executor.submit(self.compute_feature_matrix, df, X, feature_matrix)
                threads.append(t)
            executor.shutdown(wait=True)
        return feature_matrix
'''

class BioBert_en(BioBert_ru):

    def __init__(self, name, k, type, num_threads):
        super().__init__(name, k, type, num_threads)

    def create_df(self, df):
        model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")
        tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
        dfs = np.array_split(df, self.num_threads)
        feature_matrix = mp.Manager().dict()
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            for df in dfs:
                executor.submit(self.compute_feature_matrix, df, [model, tokenizer], feature_matrix)
            executor.shutdown(wait=True)
        return feature_matrix

class Smiles(Feature):

    def __init__(self, name, k, type, num_threads):
        super().__init__(name, k, type, num_threads)

    class OneHotEmbedding(nn.Module):
        def __init__(self, alphabet_size):
            super().__init__()
            self.alphabet_size = alphabet_size
            self.embedding = nn.Embedding.from_pretrained(torch.eye(alphabet_size))

        def forward(self, x):
            return self.embed(x)

    class Embedding(nn.Module):
        def __init__(self, alphabet_size, d_model):
            super().__init__()
            self.alphabet_size = alphabet_size
            self.d_model = d_model
            self.embed = nn.Embedding(alphabet_size, d_model)

        def forward(self, x):
            return self.embed(x)

    class PositionalEncoder(nn.Module):
        def __init__(self, d_model, max_seq_len=6000, dropout=0.1):
            super().__init__()
            self.d_model = d_model
            self.dropout = nn.Dropout(p=dropout)
            # create constant 'pe' matrix with values dependant on
            # pos and i
            pe = torch.zeros(max_seq_len, d_model)
            for pos in range(max_seq_len):
                for i in range(0, d_model, 2):
                    pe[pos, i] = \
                        math.sin(pos / (10000 ** ((2 * i) / d_model)))
                    pe[pos, i + 1] = \
                        math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)

        def forward(self, x):
            # make embeddings relatively larger
            x = x * math.sqrt(self.d_model)
            # add constant to embedding
            seq_len = x.size(1)
            pe = self.pe[:, :seq_len]
            pe = Variable(self.pe[:, :seq_len], requires_grad=False)
            if x.is_cuda:
                pe.cuda()
            x = x + pe
            # print(x.mean(), x)
            x = self.dropout(x)
            # x = F.dropout(x, p=0.1, training=self.training)
            # print(x.mean(), x)
            return x

    class Norm(nn.Module):
        def __init__(self, d_model, eps=1e-6):
            super().__init__()

            self.size = d_model

            # create two learnable parameters to calibrate normalisation
            self.alpha = nn.Parameter(torch.ones(self.size))
            self.bias = nn.Parameter(torch.zeros(self.size))

            self.eps = eps

        def forward(self, x):
            norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
                   / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
            return norm

    def attention(q, k, v, d_k, mask=None, dropout=None):

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)

        scores = F.softmax(scores, dim=-1)

        if dropout is not None:
            scores = dropout(scores)

        output = torch.matmul(scores, v)
        return output

    class MultiHeadAttention(nn.Module):
        def __init__(self, heads, d_model, dropout=0.1):
            super().__init__()

            self.d_model = d_model
            self.d_k = d_model // heads
            self.h = heads

            self.q_linear = nn.Linear(d_model, d_model)
            self.v_linear = nn.Linear(d_model, d_model)
            self.k_linear = nn.Linear(d_model, d_model)

            self.dropout = nn.Dropout(dropout)
            self.out = nn.Linear(d_model, d_model)

        def forward(self, q, k, v, mask=None):
            bs = q.size(0)

            # perform linear operation and split into N heads
            k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
            v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

            # transpose to get dimensions bs * N * sl * d_model
            k = k.transpose(1, 2)
            q = q.transpose(1, 2)
            v = v.transpose(1, 2)

            # calculate attention using function we will define next
            scores = Smiles.attention(q, k, v, self.d_k, mask, self.dropout)
            # concatenate heads and put through final linear layer
            concat = scores.transpose(1, 2).contiguous() \
                .view(bs, -1, self.d_model)
            output = self.out(concat)

            return output

    class FeedForward(nn.Module):
        def __init__(self, d_model, d_ff=2048, dropout=0.1):
            super().__init__()

            # We set d_ff as a default to 2048
            self.linear_1 = nn.Linear(d_model, d_ff)
            self.dropout = nn.Dropout(dropout)
            self.linear_2 = nn.Linear(d_ff, d_model)

        def forward(self, x):
            x = self.dropout(F.relu(self.linear_1(x)))
            x = self.linear_2(x)
            return x

    class EncoderLayer(nn.Module):
        def __init__(self, d_model, heads, dropout=0.1):
            super().__init__()
            self.norm_1 = Smiles.Norm(d_model)
            self.norm_2 = Smiles.Norm(d_model)
            self.attn = Smiles.MultiHeadAttention(heads, d_model, dropout=dropout)
            self.ff = Smiles.FeedForward(d_model, dropout=dropout)
            self.dropout_1 = nn.Dropout(dropout)
            self.dropout_2 = nn.Dropout(dropout)

        def forward(self, x, mask):
            x2 = self.norm_1(x)
            x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
            x2 = self.norm_2(x)
            x = x + self.dropout_2(self.ff(x2))
            return x

    # build a decoder layer with two multi-head attention layers and
    # one feed-forward layer
    class DecoderLayer(nn.Module):
        def __init__(self, d_model, heads, dropout=0.1):
            super().__init__()
            self.norm_1 = Smiles.Norm(d_model)
            self.norm_2 = Smiles.Norm(d_model)
            self.norm_3 = Smiles.Norm(d_model)

            self.dropout_1 = nn.Dropout(dropout)
            self.dropout_2 = nn.Dropout(dropout)
            self.dropout_3 = nn.Dropout(dropout)

            self.attn_1 = Smiles.MultiHeadAttention(heads, d_model, dropout=dropout)
            self.attn_2 = Smiles.MultiHeadAttention(heads, d_model, dropout=dropout)
            self.ff = Smiles.FeedForward(d_model, dropout=dropout)

        def forward(self, x, e_outputs, src_mask, trg_mask):
            x2 = self.norm_1(x)
            x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
            x2 = self.norm_2(x)
            x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, \
                                               src_mask))
            x2 = self.norm_3(x)
            x = x + self.dropout_3(self.ff(x2))
            return x

    def get_clones(module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

    class Encoder(nn.Module):
        def __init__(self, alphabet_size, d_model, N, heads, dropout):
            super().__init__()
            self.N = N
            self.embed = Smiles.Embedding(alphabet_size, d_model)
            self.pe = Smiles.PositionalEncoder(d_model, dropout=dropout)
            self.layers = Smiles.get_clones(Smiles.EncoderLayer(d_model, heads, dropout), N)
            self.norm = Smiles.Norm(d_model)

        def forward(self, src, mask):
            x = self.embed(src)
            x = self.pe(x)
            for i in range(self.N):
                x = self.layers[i](x, mask)
            return self.norm(x)

    class Decoder(nn.Module):
        def __init__(self, alphabet_size, d_model, N, heads, dropout):
            super().__init__()
            self.N = N
            self.embed = Smiles.Embedding(alphabet_size, d_model)
            self.pe = Smiles.PositionalEncoder(d_model, dropout=dropout)
            self.layers = Smiles.get_clones(Smiles.DecoderLayer(d_model, heads, dropout), N)
            self.norm = Smiles.Norm(d_model)

        def forward(self, trg, e_outputs, src_mask, trg_mask):
            x = self.embed(trg)
            x = self.pe(x)
            for i in range(self.N):
                x = self.layers[i](x, e_outputs, src_mask, trg_mask)
            return self.norm(x)

    class Transformer(nn.Module):
        def __init__(self, alphabet_size, d_model, N, heads=8, dropout=0.1):
            super().__init__()
            self.encoder = Smiles.Encoder(alphabet_size, d_model, N, heads, dropout)
            self.decoder = Smiles.Decoder(alphabet_size, d_model, N, heads, dropout)
            self.out = nn.Linear(d_model, alphabet_size)

        def forward(self, src, trg, src_mask, trg_mask):
            e_outputs = self.encoder(src, src_mask)
            # print("DECODER")
            d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
            output = self.out(d_output)
            return output

    def nopeak_mask(self, size, device):
        np_mask = torch.triu(torch.ones((size, size), dtype=torch.uint8), diagonal=1).unsqueeze(0)

        np_mask = np_mask == 0
        np_mask = np_mask.to(device)
        return np_mask

    _extra_chars = ["seq_start", "seq_end", "pad"]
    EXTRA_CHARS = {key: chr(95 + i) for i, key in enumerate(_extra_chars)}

    def create_masks(self, src, trg=None, pad_idx=ord(EXTRA_CHARS['pad']), device=None):
        src_mask = (src != pad_idx).unsqueeze(-2)

        if trg is not None:
            trg_mask = (trg != pad_idx).unsqueeze(-2)
            size = trg.size(1)  # get seq_len for matrix
            np_mask = self.nopeak_mask(size, device)
            np_mask.to(device)
            trg_mask = trg_mask & np_mask
            return src_mask, trg_mask
        return src_mask

    class CosineWithRestarts(torch.optim.lr_scheduler._LRScheduler):
        """
        Cosine annealing with restarts.
        Parameters
        ----------
        optimizer : torch.optim.Optimizer
        T_max : int
            The maximum number of iterations within the first cycle.
        eta_min : float, optional (default: 0)
            The minimum learning rate.
        last_epoch : int, optional (default: -1)
            The index of the last epoch.
        """

        def __init__(self,
                     optimizer,
                     T_max,
                     eta_min=0.,
                     last_epoch=-1,
                     factor=1.):
            # pylint: disable=invalid-name
            self.T_max = T_max
            self.eta_min = eta_min
            self.factor = factor
            self._last_restart = 0
            self._cycle_counter = 0
            self._cycle_factor = 1.
            self._updated_cycle_len = T_max
            self._initialized = False
            super(Smiles.CosineWithRestarts, self).__init__(optimizer, last_epoch)

        def get_lr(self):
            """Get updated learning rate."""
            # HACK: We need to check if this is the first time get_lr() was called, since
            # we want to start with step = 0, but _LRScheduler calls get_lr with
            # last_epoch + 1 when initialized.
            if not self._initialized:
                self._initialized = True
                return self.base_lrs

            step = self.last_epoch + 1
            self._cycle_counter = step - self._last_restart

            lrs = [
                (
                        self.eta_min + ((lr - self.eta_min) / 2) *
                        (
                                np.cos(
                                    np.pi *
                                    ((self._cycle_counter) % self._updated_cycle_len) /
                                    self._updated_cycle_len
                                ) + 1
                        )
                ) for lr in self.base_lrs
            ]

            if self._cycle_counter % self._updated_cycle_len == 0:
                # Adjust the cycle length.
                self._cycle_factor *= self.factor
                self._cycle_counter = 0
                self._updated_cycle_len = int(self._cycle_factor * self.T_max)
                self._last_restart = step

            return lrs

    def encode_char(self, c):
        return ord(c) - 32

    def encode_smiles(self, string, start_char=EXTRA_CHARS['seq_start']):
        return torch.tensor([ord(start_char)] + [self.encode_char(c) for c in string], dtype=torch.long)[
               :256].unsqueeze(0)

    ALPHABET_SIZE = 95 + len(EXTRA_CHARS)

    def compute_feature_matrix(self, df, X, feature_matrix):

        for _, row in tqdm(df.iterrows(), desc='Computing smiles feature_matrix... #rows=' + str(len(df)) +
                                                '; time: ' + datetime.now(pytz.timezone("Asia/Karachi")).strftime("%I:%M:%S%p") +
                                                '; cpu: ' + mp.current_process().name +
                                                '; thread: ' + str(threading.get_ident())):
            try:
                drug1 = row['drug1']
                drug2 = row['drug2']

                feature_vector1 = []
                feature_vector2 = []
                if drug1 in X[1]:
                    smile1 = X[1][drug1].split('\n')[0]
                    encoded1 = self.encode_smiles(smile1)
                    mask1 = self.create_masks(encoded1)
                    feature_vector1 = X[0](encoded1, mask1)[0].detach().numpy().tolist()
                    feature_vector1 = list(map(lambda x: sum(x) / len(x), zip(*feature_vector1)))
                elif len(feature_matrix) > 0:
                    feature_vector1 = np.mean([np.array(feature_matrix[key]) for key in feature_matrix], axis=0)

                if drug2 in X[1]:
                    smile2 = X[1][drug2].split('\n')[0]
                    encoded2 = self.encode_smiles(smile2)
                    mask2 = self.create_masks(encoded2)
                    feature_vector2 = X[0](encoded2, mask2)[0].detach().numpy().tolist()
                    feature_vector2 = list(map(lambda x: sum(x) / len(x), zip(*feature_vector2)))
                elif len(feature_matrix) > 0:
                    feature_vector2 = np.mean([np.array(feature_matrix[key]) for key in feature_matrix], axis=0)

                if feature_vector1 and feature_vector2:
                    feature_vector = feature_vector1 + feature_vector2
                    feature_matrix[_] = feature_vector
            except:
                continue

    def create_df(self, df):
        smiles = dict()
        with open('pubchem_smiles.csv', 'r') as file:
            file.readline()
            for line in file:
                row = line.split(',')
                drug = row[0]
                smile = row[1]
                smiles[drug] = smile
        with open('drugbank_smiles.csv', 'r') as file:
            file.readline()
            for line in file:
                row = line.split(',')
                drug = row[2]
                smile = row[4]
                if drug not in smiles:
                    smiles[drug] = smile
        model = Smiles.Transformer(Smiles.ALPHABET_SIZE, 512, 6).eval()
        model = torch.nn.DataParallel(model)
        checkpoint = torch.load('smiles.ckpt', map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint['state_dict'])
        model = model.module.cpu()
        encoder = model.encoder.cpu()
        X = [encoder, smiles]
        dfs = np.array_split(df, self.num_threads)
        feature_matrix = mp.Manager().dict()
        threads = list()
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            for df in dfs:
                t = executor.submit(self.compute_feature_matrix, df, X, feature_matrix)
                threads.append(t)
            executor.shutdown(wait=True)

        return feature_matrix

class Fasttext_ru(Feature):
    def __init__(self, name, k, type, num_threads):
        super().__init__(name, k, type, num_threads)

    def compute_feature_matrix(self, df, X, feature_matrix):
        for _, row in tqdm(df.iterrows(), desc='Computing fasttext feature_matrix... #rows=' + str(len(df)) +
                                              '; time: ' + datetime.now(pytz.timezone("Asia/Karachi")).strftime("%I:%M:%S%p") +
                                               '; cpu: ' +  mp.current_process().name +
                                               '; thread: ' + str(threading.get_ident())):
            feature_vector1 = []
            feature_vector2 = []
            try:
                feature_vector1 = list(X.get_word_vector(row['drug1']))
            except:
                pass
            try:
                feature_vector2 = list(X.get_word_vector(row['drug2']))
            except:
                pass

            if not feature_vector1 and len(feature_matrix) > 0:
                feature_vector1 = np.mean([np.array(feature_matrix[key]) for key in feature_matrix], axis=0)
            if not feature_vector2 and len(feature_matrix) > 0:
                feature_vector2 = np.mean([np.array(feature_matrix[key]) for key in feature_matrix], axis=0)

            if feature_vector1 and feature_vector2:
                feature_matrix[_] = feature_vector1 + feature_vector2

    def create_df(self, df):
        model_name = "fasttext-ru.bin"
        model = fasttext.load_model(model_name)
        dfs = np.array_split(df, self.num_threads)
        feature_matrix = mp.Manager().dict()
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            for df in dfs:
                executor.submit(self.compute_feature_matrix, df, model, feature_matrix)
            executor.shutdown(wait=True)

        return feature_matrix
