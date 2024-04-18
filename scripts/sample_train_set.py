import pandas as pd

from utils.topic_selector import *
import os

input_path = os.path.join('..', 'data', 'data_sent_annotated.csv') # input = sentence data annotated by SAS
regex_pattern = "metas|secondar|lesion|nodul|elevata attività metabolica|m\+"
output_path = os.path.join('..', 'data', 'train_data.csv')


# filter by topic the SAS kword column
topic_selector_obj = TopicSelector(input_path, regex_pattern, 'parole chiave')
data_topic = topic_selector_obj.filter_by_topic()

# sample by topic
data_tag = topic_selector_obj.tag_data_by_topic(data_topic)
data_sample = data_tag.groupby(['LEMMA', 'CATEGORY']).sample(50)[['splitted_text','id','sent_id','LEMMA','CATEGORY']]

# save train data
data_sample.to_csv(output_path, index=False)