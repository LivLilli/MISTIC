import pandas as pd

class TopicSelector():

    def __init__(self):
        pass
    def filter_by_topic(self):
        pass
    def save_topic_data(self):
        pass



df_split = df_split[
            df_split['splitted_text'].str.contains("metas") | df_split['splitted_text'].str.contains("lesion") |
            df_split['splitted_text'].str.contains("secondar") | df_split['splitted_text'].str.contains(
                "elevata attività metabolica") | df_split['splitted_text'].str.contains("m\+") | df_split[
                'splitted_text'].str.contains("nodul")]