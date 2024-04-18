from utils.data_processor import *
import os

input_path = os.path.join('..', 'data', 'data_sent_annotated.csv') # input = sentence data annotated by SAS
gs_path = os.path.join('..', 'data', 'gold_standard.csv')  # gold standard path used for selecting training sentences not in the eval set
regex_pattern = "metas|secondar|lesion|nodul|elevata attività metabolica|m\+"
output_train_path = os.path.join('..', 'data', 'train_data.csv')
output_test_path = os.path.join('..', 'data', 'test_data.csv')
output_gs_sent_path = os.path.join('..', 'data', 'gs_sent.csv')
output_gs_topic_path = os.path.join('..', 'data', 'gs_sent_filtered.csv')

processor_obj = DataProcessor(input_path, gs_path, regex_pattern)
processor_obj.save_data_for_training(output_train_path, output_test_path)
processor_obj.save_data_for_inference(output_gs_sent_path, output_gs_topic_path)