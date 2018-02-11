import cv2
import numpy as np
import skimage
import skimage.io
import re
import nltk
import tensorflow as tf
from nltk.tokenize import sent_tokenize
from html.parser import HTMLParser
from IPython.core.debugger import Tracer

def extract_argmax_and_embed(embedding, b_emb,output_projection=None,
                              update_embedding=True):
    """Get a loop_function that extracts the previous symbol and embeds it.

    Args:
      embedding: embedding tensor for symbols.
      output_projection: None or a pair (W, B). If provided, each fed previous
        output will first be multiplied by W and added B.
      update_embedding: Boolean; if False, the gradients will not propagate
        through the embeddings.

    Returns:
      A loop function.
    """
    def loop_function(prev, _):
        """function that feed previous model output rather than ground truth."""
        if output_projection is not None:
            prev = tf.nn.xw_plus_b(
                prev, output_projection[0], output_projection[1])
        prev_symbol = tf.argmax(prev, 1)
        # Note that gradients will not propagate through the second parameter of
        # embedding_lookup.
        emb_prev = tf.nn.embedding_lookup(embedding, prev_symbol) + b_emb
        if not update_embedding:
            emb_prev = tf.stop_gradient(emb_prev)
        return emb_prev
    return loop_function

def unpack_sequence(tensor):
    """Split the single tensor of a sequence into a list of frames."""
    return tf.unpack(tf.transpose(tensor)) if len(tensor.get_shape()) == 2 else tf.unpack(tf.transpose(tensor, perm=[1, 0, 2]))

def variable_summaries(name, var):
  """Attach a lot of summaries to a Tensor."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.scalar_summary('mean/' + name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.scalar_summary('stddev/' + name, stddev)
    tf.scalar_summary('max/' + name, tf.reduce_max(var))
    tf.scalar_summary('min/' + name, tf.reduce_min(var))
    tf.histogram_summary(name, var)

def tokenize(text):
    text = text.lower()
    patternComma =r'\d+[,]'
    match=re.findall(patternComma,text) #removing comma in between digits
    for item in match:
        digits=item.replace(",","")
        text=text.replace(item,digits) 

    tokens = nltk.word_tokenize(text) # something new
    tags = nltk.pos_tag(tokens)

    return tags


def extractTextFromHTML(filePath):
    #getting html and extract it to list of sentences
    html = open(filePath,"r", encoding='cp1252').read() 
    html = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', html) #remove escape characters
    text = MLStripper.strip_tags(html)

    sent_tokenize_list = sent_tokenize(text)
    text = sent_tokenize_list[0] + ' ' + sent_tokenize_list[1] + ' ' + sent_tokenize_list[2]

    return text

def read_image(path):

     img = crop_image(path, target_height=224, target_width=224)
     if img.shape[2] == 4:
         img = img[:,:,:3]

     img = img[None, ...]
     return img

def crop_image(x, target_height=227, target_width=227, as_float=True):
    image = skimage.io.imread(x)
    if as_float:
        image = skimage.img_as_float(image).astype(np.float32)

    if len(image.shape) == 2:
        image = np.tile(image[:,:,None], 3)
    elif len(image.shape) == 4:
        image = image[:,:,:,0]

    height, width, rgb = image.shape
    if width == height:
        resized_image = cv2.resize(image, (target_height,target_width))

    elif height < width:
        resized_image = cv2.resize(image, (int(width * float(target_height)/height), target_width))
        cropping_length = int((resized_image.shape[1] - target_height) / 2)
        resized_image = resized_image[:,cropping_length:resized_image.shape[1] - cropping_length]

    else:
        resized_image = cv2.resize(image, (target_height, int(height * float(target_width) / width)))
        cropping_length = int((resized_image.shape[0] - target_width) / 2)
        resized_image = resized_image[cropping_length:resized_image.shape[0] - cropping_length,:]

    return cv2.resize(resized_image, (target_height, target_width))

class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()














