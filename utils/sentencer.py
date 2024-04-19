import pandas as pd
import pysbd

class Sentencer:
    '''
    Sentencer object contains methods for process and save data to sentences.

    :param input_path: path to the input data to process
    :type input_path: str
    :param segmenter: pysbd model for segmentation
    :type segmenter: pysbd model object
    '''

    def __init__(self, input_path, segmenter):

        self.data = pd.read_csv(input_path)
        self.data_segmenter = segmenter

    def data_to_sentence(self):
        '''
        Method for segmenting input data. For using this method, dataset must contains the columns "id" and "text".

        :return: the original dataset with the new generated columns "splitted_text" and "sent_id"
        :rtype: pd.Dataframe
        '''

        data_to_split = self.data.copy()
        data_to_split['splitted_text'] = data_to_split['text'].transform(self.data_segmenter.segment)
        data_split = data_to_split.explode('splitted_text')
        output_data = data_split[['id', 'splitted_text']].dropna()
        output_data['sent_id'] = [x for x in range(1, len(output_data)+1)]

        return output_data

    def save_sentence_data(self, output_path):
        '''
        Method for saving the output segmented dataset.

        :param output_path: path where to save the output data
        :type output_path: str

        :return: saves the pandas df to a csv file
        '''

        output_data = self.data_to_sentence()
        output_data.to_csv(output_path, index=False)

