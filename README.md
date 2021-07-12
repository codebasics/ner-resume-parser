# **NER Resume Parser**


## Setup for Python:

1. Install Python ([Setup Instructions](https://wiki.python.org/moin/BeginnersGuide))

2. Install Python packages
```
pip3 install -r requirements.txt
```
3. Install MLFlow ([Setup Instructions](https://mlflow.org/docs/latest/quickstart.html))

* Tools used are : 
                **Apache Tika** - For getting pdf formatted files into to the text formatted file.
                **Doccano** - For annotating the dataset.

## Setup for Jupyter Notebook:

1. Install Everything ([Setup Instructions](https://github.com/darsh8200/ner-resume-parser/blob/main/NER-Spacy.ipynb))

## Training the Model

1. Model is Spacy:
                Training Data Set = jsonl formatted file (80 resumes used for training)
                Test Data Set = jsonl formatted file (14 resumes used for testing)

2. Blank spacy model is created with ner pipeline and disabling anyother pipeline than ner and custom labels are added into the ner pipeline.

3. -- Running .py files and flowchart for it..

4. In training shuffled and batch training concept is used for getting better training of the model.

5. In testing phase each jsonl line will be taken as input as all predictions from the text are generated into the individual text files.

## Running MLFlow

1. After training **train.py** run mlflow_spacy.py to see model on MLFlow.

