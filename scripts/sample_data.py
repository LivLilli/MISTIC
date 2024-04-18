from utils.topic_selector import *
from utils.sentencer import *
import os

input_path = os.path.join('..', 'data', 'data_sent_annotated.csv') # input = sentence data annotated by SAS
gs_path = os.path.join('..', 'data', 'gold_standard.csv')  # gold standard path used for selecting training sentences not in the eval set
regex_pattern = "metas|secondar|lesion|nodul|elevata attività metabolica|m\+"
output_train_path = os.path.join('..', 'data', 'train_data.csv')
output_test_path = os.path.join('..', 'data', 'test_data.csv')

# load gs data and save gs topic-sentences

gs_data = pd.read_csv(gs_path)
gs_text_ids = list(set(gs_data.id))

gs_sentencer = Sentencer(gs_path, pysbd.Segmenter(language="it", clean=False), '../data/gs_sent.csv')
gs_sentencer.save_sentence_data()

topic_selector_gs_obj = TopicSelector('../data/gs_sent.csv', regex_pattern, 'splitted_text', output_path='../data/gs_sent_filtered.csv')
topic_selector_gs_obj.save_topic_data()

# filter by topic the SAS kword column

topic_selector_obj = TopicSelector(input_path, regex_pattern, 'parole chiave')
data_topic = topic_selector_obj.filter_by_topic()

# sample by topic for the final train set

data_tag = topic_selector_obj.tag_data_by_topic(data_topic)
data_tag  = data_tag[~data_tag.id.isin(gs_text_ids)]# remove gs text ids
data_train = data_tag.groupby(['LEMMA', 'CATEGORY']).sample(50)[['splitted_text','id','sent_id','LEMMA','CATEGORY']]

# data for first model validation

data_test = data_tag[~data_tag.id.isin(list(set(data_train.id)))][['splitted_text','id','sent_id']].drop_duplicates()

# save train data

data_train.to_csv(output_train_path, index=False)
data_test.to_csv(output_test_path, index=False)