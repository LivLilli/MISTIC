from utils.topic_selector import *

input_path = ''
output_path = ''
regex_pattern = "metas|secondar|lesion|nodul|elevata attività metabolica|m\+"

topic_selector_obj = TopicSelector(input_path, output_path, regex_pattern)
topic_selector_obj.save_topic_data()

