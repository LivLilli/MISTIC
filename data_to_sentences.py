from utils.sentencer import *

input_path = 'raw_data/dataset.csv'
segmenter = pysbd.Segmenter(language="it", clean=False)
output_path = 'results/gold_v1_by_sents'

sentencer_obj = Sentencer(input_path, segmenter, output_path)
sentencer_obj.save_sentence_data()


