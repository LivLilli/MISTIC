# :it: :hospital: :computer: MISTIC


## Repository structure 

```bash
|-- scripts
|   |-- data_to_sentences.py
|   |-- fine_tune.py
|   |-- inference.py
|   |-- sample_data.py
|-- utils
|   |-- classifier.py
|   |-- data_processor.py
|   |-- sentencer.py
|   |-- topic_selector.py
|-- data
|   |-- data_sent_annotated.csv
|   |-- data_sent.csv
|   |-- gold_standard.csv
|   |-- gs_sent.csv
|   |-- gs_sent_filtered.csv
|   |-- input_data.csv
|   |-- test_data.csv
|   |-- train_data.csv
|-- results
|-- README.md
|-- requirements.txt
```

## How to use

1. Install requirements

```
pip install -r requirements.txt
```

2. Data Segmentation

```
python scripts/data_to_sentences.py
```

3. Sampling data for Training and Inference

```
python scripts/sample_data.py
```

4. Fine-Tuning

```
python scripts/fine_tune.py
```
5. Inference

```
python scripts/inference.py
```




