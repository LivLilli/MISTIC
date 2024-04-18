from utils.sentencer import *
import os

input_path = os.path.join('..', 'data', 'input_data.csv')
segmenter = pysbd.Segmenter(language="it", clean=False)
output_path =  os.path.join('..', 'data', 'data_sent.csv')

sentencer_obj = Sentencer(input_path, segmenter, output_path)
sentencer_obj.save_sentence_data()


