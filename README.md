# :it: :hospital: :computer: MISTIC

Welcome to MISTIC, a pipeline for Metastases Italian Sentence Transformers Inference Classification.

The research team for this work includes: Livia Lilli, Mario Santoro, Valeria Masiello, Stefano Patarnello, Luca Tagliaferri, Fabio Marazzi, Nikola Dino Capocchiano.

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
Input data is expected to have "id" and "text" columns, as follows: 

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
Training data sampling takes as input a dataset of sentences annotated by the SAS text-analytics pipeline. The dataset is required to present the columns "parole chiave" and "livello_categoria_1", where are indicated the key lemmas and the concepts of presence/absence related to those lemmas:

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

The above command performs the MISTIC fine-tuning for the given input parameters and the training data previously generated. 
The model checkpoints are saved into the "results" directory. 

5. Inference

```
python scripts/inference.py
```

The above command evaluates the fine-tuned MISTIC model on the gold standards at sentence level, previously processed in phase 3. 
The final classification is then performed at overall EHR level, by making an OR operation among the single sentences' labels. 
The output table presents the following structure:

| id | text | gold  | classification |
|----|:----:|:-----:|---------------:|
|    |      |       |                |

## Citation 

If you use MISTIC in you research, please cite this repository.
