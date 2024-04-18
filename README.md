# MISTIC

The repository present the following structure:
'''
.
├── data
│   ├── santoro_data
│   │   ├── gold_extended.csv
│   │   ├── neg_data_2024_03.csv
│   │   ├── sentences_labelled_test.csv
│   │   └── sentences_labelled_train.csv
│   ├── data_sent_annotated.csv
│   ├── data_sent.csv
│   ├── gold_standard.csv
│   ├── gs_sent.csv
│   ├── gs_sent_filtered.csv
│   ├── input_data.csv
│   ├── test_data.csv
│   └── train_data.csv
├── results
├── scripts
│   ├── data_to_sentences.py
│   ├── fine_tune.py
│   ├── inference.py
│   └── sample_data.py
├── utils
│   ├── __pycache__
│   │   ├── data_processor.cpython-310.pyc
│   │   ├── sentencer.cpython-310.pyc
│   │   └── topic_selector.cpython-310.pyc
│   ├── classifier.py
│   ├── data_processor.py
│   ├── sentencer.py
│   └── topic_selector.py
└── requirements.txt
'''

Script Order:

* data to sentences.py (tested)
* sample_data.py (tested)
* fine_tune.py (to test)
* inference.py (to test)
