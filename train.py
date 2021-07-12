import nltk
nltk.download('punkt')
nltk.download('stopwords')
#nltk.download('word_tokenize')

import json
import random
import logging
import spacy
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from spacy.gold import GoldParse
from spacy.scorer import Scorer
from sklearn.metrics import accuracy_score


LABEL = ["name", "phone_no", "email", "linkedin", "github", "designation",
         "company", "job-duration", "Experience", "degree", "academic-institute", "databases", "tools", "core-skills", "soft-skills", "cloud-platforms", 
         "Front End", "Back End", "Mobile App", "Libraries"]


training_data = []
with open("./darsh.patel.jsonl", encoding="utf8") as d:
  for line in d:
    data = json.loads(line)
    text = data["data"]
    entity = data["label"]
    entities = []
    for ent in entity:
      entities.append((ent[0], ent[1], ent[2]))
    training_data.append((text, {"entities" : entities}))
training_data

################### Train Spacy NER.###########

TRAIN_DATA = training_data[:80]
nlp = spacy.blank('en')  # create blank Language class
# create the built-in pipeline components and add them to the pipeline
# nlp.create_pipe works for built-ins that are registered with spaCy
if 'ner' not in nlp.pipe_names:
    ner = nlp.create_pipe('ner')
    nlp.add_pipe(ner, last=True)
    

for i in LABEL:
  ner.add_label(i)

# get names of other pipes to disable them during training
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
with nlp.disable_pipes(*other_pipes):  # only train NER
    optimizer = nlp.begin_training()
    for itn in range(200):
        print("Statring iteration " + str(itn))
        random.shuffle(TRAIN_DATA)
        losses = {}
        for text, annotations in TRAIN_DATA:
            nlp.update(
                [text],  # batch of texts
                [annotations],  # batch of annotations
                drop=0.3,  # dropout - make it harder to memorise data
                sgd=optimizer,  # callable to update weights
                losses=losses)
        print(losses)
