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

The above command performs EHR segmentation. 
Input data is expected to have an id and a text columns, as follows: 

| id | text |
|----|-----:| 
|    |      | 

As output, is produced a table of EHR sentences, with the following structure:
  
| id | sent_id | splitted_text |
|----|:-------:|--------------:|
|    |         |               |
   

3. Sampling data for Training and Inference

```
python scripts/sample_data.py
```

The above command filters and samples by topic the data for training. Moreover, it applies all the preprocessing pipeline (then the segmentation and the topic filtering) to the input gold standards, for being used in the inference phase.
Training data sampling takes as input a dataset of sentences annotated by the SAS text-analytics pipeline. The dataset is required to present the columns parole chiave and livello_categoria_1, where are indicated the key lemmas and the concepts of presence/absence related to those lemmas:

| id | sent_id | splitted_text | livello_categoria_1 | parole chiave |
|----|:-------:|:-------------:|:-------------------:|---------------:
|    |         |               |                     |               |

Gold standars are intended as a subset of EHR manually annotated by experts for the final model evaluation. The GS table must present the following structure: 

| id | text | gold |
|----|:----:|-----:|
|    |      |      |       


4. Fine-Tuning

```
python scripts/fine_tune.py
```

The above command performs the MISTIC fine-tuning for the given input parameters. 

5. Inference

```
python scripts/inference.py
```

The above command evaluated the fine-tuned MISTIC model on the gold standards.




