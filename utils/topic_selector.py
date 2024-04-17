import pandas as pd

class TopicSelector():

    def __init__(self, input_path, output_path, regex_pattern):

        self.data = pd.read_csv(input_path)
        self.ouput_path = output_path
        self.regex_pattern = regex_pattern
    def filter_by_topic(self):

        data_to_filter = self.data
        return data_to_filter[data_to_filter.splitted_text.str.contains(self.regex_pattern)]
    def save_topic_data(self):

        output_data = self.filter_by_topic()
        output_data.to_csv(self.ouput_path, index=False)
