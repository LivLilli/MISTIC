import pandas as pd
import numpy as np
from setfit import SetFitModel
import os
class MisticClassifier():

    def __init__(self, model_path, input_sent_data_path, input_data_path):

        self.model = SetFitModel.from_pretrained(model_path)
        self.sent_data = pd.read_csv(input_sent_data_path)
        self.data = pd.read_csv(input_data_path)
        self.output_path = output_data_path
    def classify_sentences(self):

        output_sent_data = self.sent_data.copy()
        output_sent_data['classification'] = self.model.predict(output_sent_data['splitted_text'])
        output_sent_data['classification'] = np.where(output_sent_data['classification'] < 1, 0, 1)

        return output_sent_data
    def classify_overall_ehrs(self):

        output_sent_data = self.classify_sentences()
        data = self.data.copy()

        output_sent_max_class = output_sent_data[['id', 'classification']].groupby('id').max().reset_index()
        output_data = pd.merge(data, output_sent_max_class, how='left', on='id')
        output_data['classification'] = output_data['classification'].fillna(0)

        return output_data

    def save_final_classifications(self, output_data_path):

        output_data = self.classify_overall_ehrs()
        output_data.to_csv(output_data_path)

