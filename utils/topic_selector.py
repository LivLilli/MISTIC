import pandas as pd

class TopicSelector():

    def __init__(self, input_path, regex_pattern, column_to_filter, output_path=''):

        self.data = pd.read_csv(input_path)
        self.ouput_path = output_path
        self.regex_pattern = regex_pattern
        self.column = column_to_filter
    def filter_by_topic(self):

        data_to_filter = self.data
        data_to_filter = data_to_filter[data_to_filter[self.column].notnull()]
        data_to_filter[self.column] = data_to_filter[self.column].str.lower()

        return data_to_filter[data_to_filter[self.column].str.contains(self.regex_pattern)]
    def tag_data_by_topic(self, data_filtered_by_topic):

        data_topic_tag = data_filtered_by_topic.copy()
        lemma_list = self.regex_pattern.split('|')

        for lemma in lemma_list:

            data_topic_tag.loc[data_topic_tag['parole chiave'].str.contains(lemma), 'LEMMA'] = lemma  # add lemma tag

        data_topic_tag.loc[data_topic_tag['LEMMA'] == 'secondar', 'LEMMA'] = 'metas'
        data_topic_tag.loc[data_topic_tag['livello_categoria_1'] == 'assenza_fam', 'LEMMA'] = 'familiarity'
        data_topic_tag.loc[data_topic_tag['livello_categoria_1'] == 'assenza_fam', 'CATEGORY'] = 1
        data_topic_tag.loc[data_topic_tag['livello_categoria_1'].str.contains('presenza'), 'CATEGORY'] = 1  # add lemma category tag (i.e. positive or negative lemma info)
        data_topic_tag.loc[data_topic_tag['livello_categoria_1'].str.contains('assenza'), 'CATEGORY'] = 0

        return data_topic_tag


    def save_topic_data(self):

        output_data = self.filter_by_topic()
        output_data.to_csv(self.ouput_path, index=False)
