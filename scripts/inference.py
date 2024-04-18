from utils.classifier import *

input_gs_sent_path = os.path.join('../..', 'data', 'gs_sent_filtered.csv')
input_gs_path = os.path.join('../..', 'data', 'gold_standard.csv')
output_data_path = os.path.join('../..', 'data', 'gs_classification.csv')
model_path = ''


classifier_obj = MisticClassifier(model_path, input_gs_sent_path, input_gs_path)
classifier_obj.save_final_classifications(output_data_path)


