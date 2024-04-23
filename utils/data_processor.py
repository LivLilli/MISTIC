from utils.topic_selector import *
from utils.sentencer import *

class DataProcessor:
    '''
    DataProcessor object contains methods for processing and sampling the dataset for MISTIC training and inference.

    :param input_data_path: path to the input data to process and sample
    :type input_data_path: str
    :param gs_data_path: path to the annotated gold standards
    :type gs_data_path: str
    :param regex_pattern: regex pattern used to filter texts
    :type regex_pattern: str
    :param k_sample: sampling dimension per lemma
    :type k_sample: int
    :param segmenter: pysbd segmenter
    :type segmenter: pysbd model object
    '''

    def __init__(self, input_data_path, gs_data_path, regex_pattern, k_sample, segmenter):

        self.input_data_path = input_data_path
        self.regex_pattern = regex_pattern
        self.gs_path = gs_data_path
        gs_data = pd.read_csv(gs_data_path)
        self.gs_text_ids = list(set(gs_data.id))
        self.sample_dimension = k_sample
        self.segmenter = segmenter

    def filter_data(self):
        '''
        This method leverages the TopicSelector object for fitlering and tagging input data.

        :return: dataset filtered and tagged by topic
        :rtype: pd.DataFrame
        '''

        topic_selector_obj = TopicSelector(self.input_data_path, self.regex_pattern, 'parole chiave')
        data_topic =  topic_selector_obj.filter_by_topic()

        return topic_selector_obj.tag_data_by_topic(data_topic)

    def check_data(self):
        '''
        This method filter out the gold standards from the dataset to use for training.

        :return: the input dataset, without sentences belonging to gold standard ehrs.
        '''

        data_tag = self.filter_data()

        return data_tag[~data_tag.id.isin(self.gs_text_ids)][['splitted_text', 'id', 'sent_id', 'LEMMA', 'CATEGORY']].drop_duplicates()

    def sample_train_val_data(self):
        '''
        This method sample data for training and validation of model fine-tuning.

        :return: training and validation datasets
        :rtype: tuple[pd.DataFrame, pd.DataFrame]
        '''

        data_tag_filtered = self.check_data()
        train_data =  data_tag_filtered.groupby(['LEMMA', 'CATEGORY']).sample(self.sample_dimension)
        val_data = data_tag_filtered[~data_tag_filtered.id.isin(list(set(train_data.id)))]

        return train_data, val_data

    def save_data_for_training(self, output_train_path, output_val_path):
        '''
        Method for saving the output training and validation datasets.

        :param output_train_path: path where to save the training data
        :type output_train_path: str
        :param output_val_path: path where to save the validation data
        :type output_val_path: str

        :return: saves the pandas df to a csv file
        '''

        train_data, val_data = self.sample_train_val_data()
        train_data.to_csv(output_train_path, index=False)
        val_data.to_csv(output_val_path, index=False)

    def save_data_for_inference(self, output_gs_sent_path, output_gs_sent_filtered_path):
        '''
        Method for saving the gs dataset for inference.

        :param output_gs_sent_path: path where to save the gs sentence data
        :type output_gs_sent_path: str
        :param output_gs_sent_filtered_path: path where to save the gs sentence data filtered by topic
        :type output_gs_sent_filtered_path: str

        :return: saves the pandas df to a csv file
        '''

        gs_sentencer = Sentencer(self.gs_path, self.segmenter)
        gs_sentencer.save_sentence_data(output_gs_sent_path)
        topic_selector_gs_obj = TopicSelector(output_gs_sent_path, self.regex_pattern, 'splitted_text')
        topic_selector_gs_obj.save_topic_data(output_gs_sent_filtered_path)

