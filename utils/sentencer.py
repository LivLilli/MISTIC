import pandas as pd
import pysbd
class Sentencer():

    def __init__(self, input_path, segmenter):

        self.data = pd.read_csv(input_path)
        self.data_segmenter = segmenter
    def data_to_sentence(self):

        data_to_split = self.data.copy()
        data_to_split['splitted_text'] = data_to_split['text'].transform(self.data_segmenter.segment)
        data_split = data_to_split.explode('splitted_text')
        output_data = data_split[['id', 'splitted_text']].dropna()
        output_data['sent_id'] = [x for x in range(1, len(output_data)+1)]

        return output_data

    def save_sentence_data(self, output_path):

        output_data = self.data_to_sentence()
        output_data.to_csv(output_path, index=False)

