#-*- coding: utf-8 -*-
import math
import os
import glob
import tensorflow as tf
import numpy as np
from IPython.core.debugger import Tracer
import pandas as pd
import skimage
from skimage import *
import pickle #python 2.7
import cv2
import seq2seq_lib
from tensorflow.python.ops import rnn_cell
import tensorflow.python.platform
from keras.preprocessing import sequence
from collections import Counter
from asai_util import *

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('iter', '68',
                           'iter')

################### Parameters #####################x
num_word_count_threshold=7
val_percent = 0.05
test_percent = 0.05
dim_embed = 128
dim_hidden = 128
dim_image = 4096
maxlen=50
model_path = './models/model-'+FLAGS.iter
data_path = './data/'
################################################################
class Caption_Generator():

    def init_weight(self, dim_in, dim_out, name=None, stddev=1.0):
        return tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=stddev / math.sqrt(float(dim_in))), name=name)

    def init_bias(self, dim_out, name=None):
        return tf.Variable(tf.zeros([dim_out]), name=name)

    def __init__(self, dim_image, dim_embed, dim_hidden, batch_size, n_lstm_steps, n_words, enc_timesteps, bias_init_vector=None):

        self.dim_image = np.int(dim_image)
        self.dim_embed = np.int(dim_embed)
        self.dim_hidden = np.int(dim_hidden)
        self.batch_size = np.int(batch_size)
        self.n_lstm_steps = np.int(n_lstm_steps)
        self.n_words = np.int(n_words)
        self.enc_timesteps = np.int(enc_timesteps)
        with tf.device("/cpu:0"):
            self.Wemb = tf.Variable(tf.random_uniform(
                [n_words, dim_embed], -0.1, 0.1), name='Wemb')

        self.bemb = self.init_bias(dim_embed, name='bemb')

        self.lstm = rnn_cell.LSTMCell(dim_hidden, state_is_tuple=True)
        self.lstm = rnn_cell.DropoutWrapper(self.lstm, input_keep_prob=1)
        self.lstm = rnn_cell.MultiRNNCell([self.lstm ])

        self.back_lstm = rnn_cell.LSTMCell(dim_hidden, state_is_tuple=True)
        self.back_lstm = rnn_cell.DropoutWrapper(self.back_lstm, input_keep_prob=1)
        self.back_lstm = rnn_cell.MultiRNNCell([self.back_lstm])
        self.encode_img_W = tf.Variable(tf.random_uniform(
            [dim_image, dim_hidden], -0.1, 0.1), name='encode_img_W')
        self.encode_img_b = self.init_bias(dim_hidden, name='encode_img_b')

        self.embed_word_W = tf.Variable(tf.random_uniform(
            [dim_hidden, n_words], -0.1, 0.1), name='embed_word_W')

        if bias_init_vector is not None:
            self.embed_word_b = tf.Variable(
                bias_init_vector.astype(np.float32), name='embed_word_b')
        else:
            self.embed_word_b = self.init_bias(n_words, name='embed_word_b')

    def build_generator(self, maxlen):
        image = tf.placeholder(tf.float32, [1, self.dim_image], name='image')
        image_emb = tf.matmul(image, self.encode_img_W) + \
            self.encode_img_b
        captions = tf.placeholder(
            tf.int32, [1, self.n_lstm_steps])
        articles = tf.placeholder(
            tf.int32, [1, None], name='articles')
        news_len = tf.placeholder(tf.int32, [1])

        mask = tf.placeholder(tf.float32, [1, self.n_lstm_steps], name='news_len')

        state = self.lstm.zero_state(1, tf.float32)

        generated_words = []

        loss = 0.0
        with tf.variable_scope("encoder"):
            current_emb = tf.nn.embedding_lookup(
                self.Wemb, articles) + self.bemb
            current_emb = tf.concat( #for image
                1, [tf.expand_dims(image_emb, 1), current_emb])
            encoder_outputs, state = tf.nn.bidirectional_dynamic_rnn(
                self.lstm, self.back_lstm, current_emb, news_len, dtype=tf.float32)
            state = state[0]

            encoder_outputs = tf.concat(1, encoder_outputs)

        with tf.variable_scope("decoder"):
            loop_function = extract_argmax_and_embed(
                self.Wemb, self.bemb, (self.embed_word_W, self.embed_word_b), update_embedding=False)
            
            current_emb = tf.nn.embedding_lookup(
                self.Wemb, captions) + self.bemb
            current_emb = unpack_sequence(current_emb)
            cell = tf.nn.rnn_cell.LSTMCell(
                dim_hidden,
                state_is_tuple=True,
                initializer=tf.random_uniform_initializer(
                    -0.1, 0.1, seed=113))
            cell = rnn_cell.DropoutWrapper(self.lstm, output_keep_prob=1)

            decoder_outputs, dec_out_state = tf.nn.seq2seq.attention_decoder(
                decoder_inputs=current_emb,
                initial_state=state,
                attention_states=encoder_outputs,
                cell=cell,
                output_size=None,
                num_heads=1,
                loop_function=loop_function,
                dtype=None, scope=None, initial_state_attention=True)

        model_outputs = []
        with tf.variable_scope("loss"):
            for i in range(1,self.n_lstm_steps):  # maxlen + 1
                output = decoder_outputs[i]

                labels = tf.expand_dims(captions[:, i], 1)  # (batch_size)
                indices = tf.expand_dims(
                    tf.range(0, 1, 1), 1)
                concated = tf.concat(1, [indices, labels])
                onehot_labels = tf.sparse_to_dense(
                    concated, tf.pack([1, self.n_words]), 1.0, 0.0)  # (batch_size, n_words)

                # (batch_size, n_words)
                logit_words = tf.matmul(
                    output, self.embed_word_W) + self.embed_word_b
                max_prob_word = tf.argmax(logit_words, 1)
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                    logit_words, onehot_labels)
                cross_entropy = cross_entropy * \
                    mask[:, i]  # tf.expand_dims(mask, 1)

                current_loss = tf.reduce_sum(cross_entropy)
                loss = loss + current_loss

                model_outputs.append(max_prob_word)

            loss = loss / tf.reduce_sum(mask[:, 1:])

        return image, model_outputs, articles, news_len, loss, captions, mask


def get_caption_data():
    feats = np.load(data_path + 'img-cnn-dataset.npy') 
    feats = np.reshape(feats,(-1,dim_image))
    captions = np.load(data_path + 'caption-dataset.npy', encoding='bytes') # -dm-2-1sent
    articles = np.load(data_path + 'article-dataset.npy', encoding='bytes') 
    return feats, captions, articles


def read_image(path):

    img = crop_image(path, target_height=224, target_width=224)
    if img.shape[2] == 4:
        img = img[:, :, :3]

    img = img[None, ...]
    return img

def test():
    def captioning(test_image_path=None, test_html_path=None, txtPath=None):
        image_val = read_image(test_image_path)
        fc7 = sess.run(graph.get_tensor_by_name("import/fc7_relu:0"), feed_dict={images:image_val})
        one_sentence = extractTextFromHTML(test_html_path)

        sentence = open(txtPath, "r", 'cp1252')
        caption = sentence.readline()
        caption = caption.replace('\n', ' .')

        return captioning_ready(fc7, one_sentence, caption)
    

    def captioning_ready(fc7, one_sentence, caption):
        current_caption_ind = []
        words = tokenize(caption)[:-1]# sentence.lower().split(' ')[:-1]
        for word in words:           
            news_word = wordtoix[word[0]] if word[0] in wordtoix else wordtoix['#UNK#']
            current_caption_ind.append(news_word)
        
        current_caption_matrix = sequence.pad_sequences(
            [current_caption_ind], padding='post', maxlen=maxlen-1)
        current_caption_matrix = np.hstack(
            [np.full((len(current_caption_matrix), 1), 0), current_caption_matrix]).astype(int)


        current_mask_matrix = np.zeros(
            (current_caption_matrix.shape[0], current_caption_matrix.shape[1]))
        nonzeros = np.array(
            [(x != 0).sum() + 2 for x in current_caption_matrix])
        for ind, row in enumerate(current_mask_matrix):
            row[:nonzeros[ind]] = 1

        #dealing with news
        current_news_ind = []
        words = tokenize(one_sentence)# sentence.lower().split(' ')[:-1]
        for word in words:  
            news_word = wordtoix[word[0]] if word[0] in wordtoix else wordtoix['#UNK#']
            current_news_ind.append(news_word)
        current_news_matrix = [np.concatenate([[0],current_news_ind,[0]])]
        current_news_len = [len(current_news_ind)]

        generated_word_index, lossnya = sess.run([generated_words_tf, loss], feed_dict=
                                       {fc7_tf:fc7,
                                        news_sentence: current_news_matrix, 
                                        news_len: current_news_len,
                                        caption_model: current_caption_matrix,
                                        mask: current_mask_matrix
                                       })
        generated_word_index = np.hstack(generated_word_index)
        generated_words = [ixtoword[x] for x in generated_word_index]
        punctuation = np.argmax(np.array(generated_words) == '#START#')+0 if '#START#' in generated_words else len(generated_words)-1
        
        generated_words = generated_words[:punctuation]
        return generated_words

    ixtoword = np.load('./data/ixtoword_DM2_7.npy').tolist() 
    wordtoix = np.load('./data/wordtoix_DM2_7.npy').tolist()
    n_words = len(ixtoword)

    # CPU mode
    config = tf.ConfigProto(
        device_count={'GPU': 0}
    )
    sess = tf.InteractiveSession(config=config)
    # sess = tf.InteractiveSession()
     
    caption_generator = Caption_Generator(
            dim_image=dim_image,
            dim_hidden=dim_hidden,
            dim_embed=dim_embed,
            batch_size=1,
            n_lstm_steps=maxlen,
            enc_timesteps=4000 + 2,
            n_words=n_words)

    graph = tf.get_default_graph()

    fc7_tf, generated_words_tf, news_sentence, news_len, loss, caption_model, mask = caption_generator.build_generator(maxlen=maxlen)

    saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
    saver.restore(sess, model_path)

    feats, captions, sentencesList = get_caption_data()
    test_index = np.load('./data/test_index.npy')

    feats = feats[test_index]
    captions = captions[test_index]
    sentencesList = sentencesList[test_index]

    total_BLEU_1 = 0
    total_BLEU_2 = 0
    total_BLEU_3 = 0
    total_BLEUscore_sentence = 0

    # num_iter = 100
    num_iter = len(feats) #len(feats)
    generated_sentences = []
    

    for index in range(0,num_iter):
        print(index)
        hypothesis = captioning_ready(np.expand_dims(feats[index], axis=0),sentencesList[index],captions[index])
        hypothesis = [word.lower() for word in hypothesis]
        generated_sentence = ' '.join(hypothesis)

        curses = set(['``',"'","''"])
        hypothesis = list(filter(lambda x: x not in curses, hypothesis))

        reference = nltk.word_tokenize(captions[index])
        reference = [word.lower() for word in reference]   

        reference = list(filter(lambda x: x not in curses, reference))

        BLEUscore_sentence = 0
        BLEUscore_1 = 0
        BLEUscore_2 = 0
        BLEUscore_3 = 0

        try:
            BLEUscore_sentence = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis)
            BLEUscore_1 = float(nltk.translate.bleu_score.modified_precision([reference], hypothesis, n=1))
            BLEUscore_2 = float(nltk.translate.bleu_score.modified_precision([reference], hypothesis, n=2))
            BLEUscore_3 = float(nltk.translate.bleu_score.modified_precision([reference], hypothesis, n=3))
        except:
            pass
        
        generated_sentences.append(generated_sentence)

        total_BLEUscore_sentence += BLEUscore_sentence
        total_BLEU_1 += BLEUscore_1
        total_BLEU_2 += BLEUscore_2
        total_BLEU_3 += BLEUscore_3
        
        print('\n')

    print('TOTAL BLEU')
    print(total_BLEUscore_sentence/num_iter)
    print(total_BLEU_1/num_iter)
    print(total_BLEU_2/num_iter)
    print(total_BLEU_3/num_iter)


    np.save('./generated_sentences.npy', generated_sentences)
    np.save('./test_captions.npy',captions)
    np.save('./sentencesList.npy',sentencesList)

def main(unused_argv):
    test()

if __name__ == '__main__':
  tf.app.run()
