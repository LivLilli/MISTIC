import pandas as pd
import numpy as np
from setfit import SetFitModel
import os

class MisticClassifier:
    '''
    MisticClassifier object classify model though the input model wrt the metastasis task.

    :param model_path: path to the model classifier checkpoints
    :type model_path: str
    :param  input_sent_data_path: path to the sentence data to be classified
    :type input_sent_data_path: str
    :param input_data_path: path to the input entire ehr data for final classification
    :type input_data_path: str
    '''

    def __init__(self, model_path, input_sent_data_path, input_data_path):

        self.model = SetFitModel.from_pretrained(model_path)
        self.sent_data = pd.read_csv(input_sent_data_path)
        self.data = pd.read_csv(input_data_path)

    def classify_sentences(self):
        '''
        This method classifies sentences wrt to the binary outcome. Sentence dataset must contains the columns
        "splitted_text" and "id" for using this function.

        :return: original dataset with the new "classification" column
        :rtype: pd.DataFrame
        '''

        output_sent_data = self.sent_data.copy()
        output_sent_data['classification'] = self.model.predict(output_sent_data['splitted_text'])
        output_sent_data['classification'] = np.where(output_sent_data['classification'] < 1, 0, 1)

        return output_sent_data

    def classify_overall_ehrs(self):
        '''
        This method classifies the overall ehrs making an OR among the sentence-predictions. The ehr dataset must contain
        the "id" column for using this function.

        :return: the original ehr dataset with the new "classification" column
        :rtype: pd.DataFrame
        '''

        output_sent_data = self.classify_sentences()
        data = self.data.copy()

        output_sent_max_class = output_sent_data[['id', 'classification']].groupby('id').max().reset_index()
        output_data = pd.merge(data, output_sent_max_class, how='left', on='id')
        output_data['classification'] = output_data['classification'].fillna(0)

        return output_data

    def save_final_classifications(self, output_data_path):
        '''
        Method for saving the classified ehr dataset.

        :param output_path: path where to save the output data
        :type output_path: str

        :return: saves the pandas df to a csv file
        '''

        output_data = self.classify_overall_ehrs()
        output_data.to_csv(output_data_path, index=False)

