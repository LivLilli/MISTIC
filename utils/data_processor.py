from utils.topic_selector import *
from utils.sentencer import *
class DataProcessor:

    def __init__(self, input_data_path, gs_data_path, regex_pattern):

        self.input_data_path = input_data_path
        self.regex_pattern = regex_pattern
        self.gs_path = gs_data_path
        gs_data = pd.read_csv(gs_data_path)
        self.gs_text_ids = list(set(gs_data.id))

    def filter_data(self):

        topic_selector_obj = TopicSelector(self.input_data_path, self.regex_pattern, 'parole chiave')
        data_topic =  topic_selector_obj.filter_by_topic()

        return topic_selector_obj.tag_data_by_topic(data_topic)

    def check_data(self):

        data_tag = self.filter_data()

        return data_tag[~data_tag.id.isin(self.gs_text_ids)][['splitted_text', 'id', 'sent_id', 'LEMMA', 'CATEGORY']].drop_duplicates()

    def sample_train_val_data(self):

        data_tag_filtered = self.check_data()
        train_data =  data_tag_filtered.groupby(['LEMMA', 'CATEGORY']).sample(50)
        val_data = data_tag_filtered[~data_tag_filtered.id.isin(list(set(train_data.id)))]

        return train_data, val_data

    def save_data_for_training(self, output_train_path, output_val_path):

        train_data, val_data = self.sample_train_val_data()
        train_data.to_csv(output_train_path, index=False)
        val_data.to_csv(output_val_path, index=False)
    def save_data_for_inference(self, output_gs_sent_path, output_gs_sent_filtered_path):

        gs_sentencer = Sentencer(self.gs_path, pysbd.Segmenter(language="it", clean=False))
        gs_sentencer.save_sentence_data(output_gs_sent_path)

        topic_selector_gs_obj = TopicSelector(output_gs_sent_path, self.regex_pattern, 'splitted_text')
        topic_selector_gs_obj.save_topic_data(output_gs_sent_filtered_path)

