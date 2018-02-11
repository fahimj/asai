#-*- coding: utf-8 -*-
import math
import os
import glob
import tensorflow as tf
import numpy as np
from IPython.core.debugger import Tracer
from skimage import *
import pickle #python 2.7
import cv2
import seq2seq_lib
import shutil

from tensorflow.python.ops import rnn_cell
from keras.preprocessing import sequence
from asai_util import *

################### Parameters ########################
dim_image = 4096
dropout = 0.30
n_epochs = 1000

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('num_word_count_threshold', 7,
                            'num_word_count_threshold')
tf.app.flags.DEFINE_integer('num_rnn_layers', 1,
                            'num_rnn_layers')
tf.app.flags.DEFINE_integer('dim_embed', 32,
                            'dim_embed')
tf.app.flags.DEFINE_integer('dim_hidden', 32,
                            'dim_hidden')
tf.app.flags.DEFINE_integer('batch_size', 24,
                            'batch_size')
tf.app.flags.DEFINE_float('dropout', 1,
                            'dropout')

root_path = os.getcwd()
data_path = './data/'

##############################################################
class Caption_Generator():

    def init_weight(self, dim_in, dim_out, name=None, stddev=1.0):
        return tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=stddev / math.sqrt(float(dim_in))), name=name)

    def init_bias(self, dim_out, name=None):
        return tf.Variable(tf.zeros([dim_out]), name=name)

    def __init__(self, dim_image, dim_embed, dim_hidden, batch_size, n_lstm_steps, n_words, enc_timesteps, bias_init_vector=None):

        self.dim_image = np.int(dim_image)
        self.dim_embed = np.int(FLAGS.dim_embed)
        self.dim_hidden = np.int(FLAGS.dim_hidden)
        self.batch_size = np.int(FLAGS.batch_size)
        self.n_lstm_steps = np.int(n_lstm_steps)
        self.n_words = np.int(n_words)
        self.enc_timesteps = np.int(enc_timesteps)
        with tf.device("/cpu:0"):
            self.Wemb = tf.Variable(tf.random_uniform(
                [n_words, FLAGS.dim_embed], -0.1, 0.1), name='Wemb')

        self.bemb = self.init_bias(FLAGS.dim_embed, name='bemb')

        self.lstm = rnn_cell.LSTMCell(FLAGS.dim_hidden, state_is_tuple=True)
        self.lstm = rnn_cell.DropoutWrapper(self.lstm, input_keep_prob=FLAGS.dropout)
        self.lstm = rnn_cell.MultiRNNCell([self.lstm ] * FLAGS.num_rnn_layers)

        self.back_lstm = rnn_cell.LSTMCell(FLAGS.dim_hidden, state_is_tuple=True)
        self.back_lstm = rnn_cell.DropoutWrapper(self.back_lstm, input_keep_prob=FLAGS.dropout)
        self.back_lstm = rnn_cell.MultiRNNCell([self.back_lstm] * FLAGS.num_rnn_layers)

        self.encode_img_W = tf.Variable(tf.random_uniform(
            [dim_image, FLAGS.dim_hidden], -0.1, 0.1), name='encode_img_W')
        self.encode_img_b = self.init_bias(FLAGS.dim_hidden, name='encode_img_b')

        self.embed_word_W = tf.Variable(tf.random_uniform(
            [FLAGS.dim_hidden, n_words], -0.1, 0.1), name='embed_word_W')

        if bias_init_vector is not None:
            self.embed_word_b = tf.Variable(
                bias_init_vector.astype(np.float32), name='embed_word_b')
        else:
            self.embed_word_b = self.init_bias(n_words, name='embed_word_b')

    def build_model(self):
        image = tf.placeholder(tf.float32, [self.batch_size, self.dim_image])
        image_emb = tf.matmul(image, self.encode_img_W) + \
            self.encode_img_b
        captions = tf.placeholder(
            tf.int32, [self.batch_size, self.n_lstm_steps], name='captions')
        articles = tf.placeholder(
            tf.int32, [self.batch_size, None], name='articles')# self.enc_timesteps])
        news_len = tf.placeholder(tf.int32, [self.batch_size], name='news_len')

        mask = tf.placeholder(tf.float32, [self.batch_size, self.n_lstm_steps])

        state = self.lstm.zero_state(self.batch_size, tf.float32)

        loss = 0.0
        with tf.variable_scope("encoder"):
            # Dealing with news text
            current_emb = tf.nn.embedding_lookup(
                self.Wemb, articles) + self.bemb
            current_emb = tf.concat( #for image
                1, [tf.expand_dims(image_emb, 1), current_emb])
            encoder_outputs, state = tf.nn.bidirectional_dynamic_rnn(
                self.lstm, self.back_lstm, current_emb, news_len, dtype=tf.float32)
            state = state[0]
            encoder_outputs = tf.concat(1, encoder_outputs)

        with tf.variable_scope("decoder"):

            current_emb = tf.nn.embedding_lookup(
                self.Wemb, captions) + self.bemb
            current_emb = unpack_sequence(current_emb)
            cell = tf.nn.rnn_cell.LSTMCell(
                FLAGS.dim_hidden,
                state_is_tuple=True,
                initializer=tf.random_uniform_initializer(
                    -0.1, 0.1, seed=113))#,
            cell = rnn_cell.DropoutWrapper(self.lstm, output_keep_prob=FLAGS.dropout)
            decoder_outputs, dec_out_state = tf.nn.seq2seq.attention_decoder(
                decoder_inputs=current_emb,
                initial_state=state,
                attention_states=encoder_outputs,
                cell=cell,
                output_size=None,
                num_heads=1, dtype=None, scope=None, initial_state_attention=False)

        with tf.variable_scope('loss'):
            def sampled_loss_func(inputs, labels):
                with tf.device('/cpu:0'):  # Try gpu.
                    labels = tf.reshape(labels, [-1, 1])
                    return tf.nn.sampled_softmax_loss(tf.transpose(self.embed_word_W), self.embed_word_b, inputs, labels,
                        4096, self.n_words) #4096

            decoder_outputs = decoder_outputs[:-1]
            sentence_modif = tf.slice(captions, [0, 1], [-1, -1])
            mask_modif = tf.slice(mask, [0,0],[-1,self.n_lstm_steps-1],name='mask')
            loss = seq2seq_lib.sampled_sequence_loss(
                decoder_outputs, unpack_sequence(sentence_modif), unpack_sequence(mask_modif), sampled_loss_func)
            variable_summaries("loss", loss)

        with tf.variable_scope('output'):
            model_outputs = []
            for i in range(len(decoder_outputs)):
                model_outputs.append(
                    tf.nn.xw_plus_b(decoder_outputs[i], self.embed_word_W, self.embed_word_b))

        with tf.variable_scope('decode_output'), tf.device('/cpu:0'):
            best_outputs = [tf.argmax(x, 1) for x in model_outputs]
            best_outputs = tf.transpose(best_outputs)

        return loss, image, captions, mask, articles, news_len

def get_caption_data():
    feats = np.load(data_path + 'img-cnn-dataset.npy') 
    feats = np.reshape(feats,(-1,dim_image))
    captions = np.load(data_path + 'caption-dataset.npy', encoding='bytes') # -dm-2-1sent
    articles = np.load(data_path + 'article-dataset.npy', encoding='bytes') 

    return feats, captions, articles

# borrowed this function from NeuralTalk
def preProBuildWordVocab(sentence_iterator, word_count_threshold=FLAGS.num_word_count_threshold):
    print('preprocessing word counts and creating vocab based on word count threshold %d' % (
        word_count_threshold, ))
    word_counts = {}
    nsents = 0
    for sent in sentence_iterator:
        nsents += 1
        for w in tokenize(sent):# sent.lower().split(' '):
            word_counts[w[0]] = word_counts.get(w[0], 0) + 1
    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print('filtered words from %d to %d' % (len(word_counts), len(vocab)))

    ixtoword = {}
    ixtoword[0] = '#START#'
    ixtoword[1] = '#UNK#' #unknown token
    wordtoix = {}
    wordtoix['#START#'] = 0  # make first vector be the start token
    wordtoix['#UNK#'] = 1
    ix = 2
    for w in vocab:
        wordtoix[w] = ix
        ixtoword[ix] = w
        
        ix += 1

    word_counts['#START#'] = nsents
    word_counts['#UNK#'] = 1

    bias_init_vector = np.array(
        [1.0 * word_counts[ixtoword[i]] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector)  # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector)  # shift to nice numeric range

    return wordtoix, ixtoword, bias_init_vector

def train():
    shutil.rmtree(os.path.join(root_path,"./logs/train_logs"),ignore_errors=True) #remove previous log files
    shutil.rmtree(os.path.join(root_path,"./logs/val_logs"),ignore_errors=True)

    learning_rate = 5e-4# 0.001
    feats, captions, sentencesList = get_caption_data()

    if (os.path.isfile(os.path.join(root_path, 'data/ixtoword_DM2_7.npy'))): #ixtoword_DM2_5_1sent
        ixtoword = np.load(os.path.join(root_path, 'data/ixtoword_DM2_7.npy'))[()]
        wordtoix = np.load(os.path.join(root_path, 'data/wordtoix_DM2_7.npy'))[()]

        bias_init_vector = np.load(os.path.join(root_path, 'data/bias_init_vector_7.npy'))[()]

    else:
        wordtoix, ixtoword, bias_init_vector = preProBuildWordVocab(sentencesList)
        np.save(os.path.join(root_path, 'data/ixtoword_DM2_7'), ixtoword)
        np.save(os.path.join(root_path, 'data/wordtoix_DM2_7'), wordtoix)
        np.save(os.path.join(root_path, 'data/bias_init_vector_7'), bias_init_vector)

    train_index = np.load('./data/train_index.npy')
    val_index = np.load('./data/val_index.npy')

    feats_val = feats[val_index]
    captions_val = captions[val_index]
    sentencesList_val = sentencesList[val_index]

    feats = feats[train_index]
    captions = captions[train_index]
    sentencesList = sentencesList[train_index]

    # CPU mode
    config = tf.ConfigProto(
        device_count={'GPU': 0}
    )

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth=True

    sess = tf.InteractiveSession(config=config)

    n_words = len(wordtoix)
    maxlen = np.max([len(x.split(' ')) for x in captions])  # the longest length of captions in training data
   
    maxlenNews = np.max([len(x.split(' ')) for x in sentencesList])
    caption_generator = Caption_Generator(
        dim_image=dim_image,
        dim_hidden=FLAGS.dim_hidden,
        dim_embed=FLAGS.dim_embed,
        batch_size=FLAGS.batch_size,
        n_lstm_steps=maxlen + 2,
        n_words=n_words,
        bias_init_vector=bias_init_vector,
        enc_timesteps=maxlenNews + 2)

    loss, image, sentence, mask, news_sentence, news_len = caption_generator.build_model()
   
    saver = tf.train.Saver(max_to_keep=5000, write_version=tf.train.SaverDef.V2)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    writer = tf.train.SummaryWriter(os.path.join(root_path,"./logs/train_logs"), sess.graph) # for 0.8
    writer_val = tf.train.SummaryWriter(os.path.join(root_path,"./logs/val_logs"))

    merged = tf.merge_all_summaries()

    tf.initialize_all_variables().run()

    for epoch in range(n_epochs):

        #training
        for start, end in zip(
                list(range(0, len(feats), FLAGS.batch_size)),
                list(range(FLAGS.batch_size, len(feats), FLAGS.batch_size))
        ):

            current_feats = feats[start:end]
            current_captions = captions[start:end]
            current_news = sentencesList[start:end]

            # dealing with news
            current_news_ind = []
            for one_sentence in current_news:
                temp = []
                words = tokenize(one_sentence)# sentence.lower().split(' ')[:-1]
                for word in words:
                    news_word = wordtoix[word[0]] if word[0] in wordtoix else wordtoix['#UNK#']
                    temp.append(news_word)
                current_news_ind.append(temp)

            current_caption_ind = []
            for one_sentence in current_captions:
                temp = []
                words = tokenize(one_sentence)[:-1]# sentence.lower().split(' ')[:-1]
                for word in words:
                    news_word = wordtoix[word[0]] if word[0] in wordtoix else wordtoix['#UNK#']
                    temp.append(news_word)
                current_caption_ind.append(temp)

            current_news_len = [len(x)+1 for x in current_news_ind]
            max_current_news = max(current_news_len) # + 1

            current_news_matrix = sequence.pad_sequences(
                current_news_ind, padding='post', maxlen=max_current_news)
            current_news_matrix = np.hstack(
                [np.full((len(current_news_matrix), 1), 0), current_news_matrix]).astype(int)

            current_caption_matrix = sequence.pad_sequences(
                current_caption_ind, padding='post', maxlen=maxlen + 1)
            current_caption_matrix = np.hstack(
                [np.full((len(current_caption_matrix), 1), 0), current_caption_matrix]).astype(int)

            current_mask_matrix = np.zeros(
                (current_caption_matrix.shape[0], current_caption_matrix.shape[1]))
            nonzeros = np.array(
                [(x != 0).sum() + 2 for x in current_caption_matrix])
            #  +2 -> #START# and 'unk'

            for ind, row in enumerate(current_mask_matrix):
                row[:nonzeros[ind]] = 1

            _, loss_value, summary = sess.run([train_op, loss, merged], feed_dict={
                image: current_feats,
                sentence: current_caption_matrix,
                mask: current_mask_matrix,
                news_sentence: current_news_matrix,
                news_len: current_news_len
            })

            print("Current Cost: ", loss_value)

        #validation
        for start, end in zip(
                list(range(0, len(feats_val), FLAGS.batch_size)),
                list(range(FLAGS.batch_size, len(feats_val), FLAGS.batch_size))
        ):

            current_feats = feats_val[start:end]
            current_captions = captions_val[start:end]
            current_news = sentencesList_val[start:end]

            # dealing with news
            current_news_ind = []
            for one_sentence in current_news:
                temp = []
                words = tokenize(one_sentence)
                for word in words:
                    news_word = wordtoix[word[0]] if word[0] in wordtoix else wordtoix['#UNK#']
                    temp.append(news_word)
                current_news_ind.append(temp)

            current_caption_ind = []
            for one_sentence in current_captions:
                temp = []
                words = tokenize(one_sentence)[:-1]
                for word in words:
                    news_word = wordtoix[word[0]] if word[0] in wordtoix else wordtoix['#UNK#']
                    temp.append(news_word)
                current_caption_ind.append(temp)

            current_news_len = [len(x)+1 for x in current_news_ind]
            max_current_news = max(current_news_len)

            current_news_matrix = sequence.pad_sequences(
                current_news_ind, padding='post', maxlen=max_current_news)
            current_news_matrix = np.hstack(
                [np.full((len(current_news_matrix), 1), 0), current_news_matrix]).astype(int)


            current_caption_matrix = sequence.pad_sequences(
                current_caption_ind, padding='post', maxlen=maxlen + 1)
            current_caption_matrix = np.hstack(
                [np.full((len(current_caption_matrix), 1), 0), current_caption_matrix]).astype(int)

            current_mask_matrix = np.zeros(
                (current_caption_matrix.shape[0], current_caption_matrix.shape[1]))
            nonzeros = np.array(
                [(x != 0).sum() + 2 for x in current_caption_matrix])

            for ind, row in enumerate(current_mask_matrix):
                row[:nonzeros[ind]] = 1

            loss_value, summary_val = sess.run([loss, merged], feed_dict={
                image: current_feats,
                sentence: current_caption_matrix,
                mask: current_mask_matrix,
                news_sentence: current_news_matrix,
                news_len: current_news_len
            })
            print("Current Validation Cost: ", loss_value)
            
        print("Epoch ", epoch, " is done. Saving the model ... ")
        writer.add_summary(summary, epoch)  # Write summary
        writer_val.add_summary(summary_val, epoch)
        saver.save(sess, os.path.join(root_path,'models'), global_step=epoch)
        learning_rate *= 0.95

def main(unused_argv):
    train()

if __name__ == '__main__':
  tf.app.run()
