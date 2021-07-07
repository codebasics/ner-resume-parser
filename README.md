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

1. Install Python packages
```
!pip3 install -r requirements.txt
```
``` bash
import mlflow
with mlflow.start_run(run_name="MLflow on Colab"):
  mlflow.log_metric("m1", 2.0)
  mlflow.log_param("p1", "mlflow-colab")
# run tracking UI in the background
get_ipython().system_raw("mlflow ui --port 5000 &")

from pyngrok import ngrok
from getpass import getpass
# Terminate open tunnels if any exist
ngrok.kill()

# Enter your auth token when the code is running
NGROK_AUTH_TOKEN = getpass('Enter the ngrok authtoken: ')
ngrok.set_auth_token(NGROK_AUTH_TOKEN)
ngrok_tunnel = ngrok.connect(addr="5000", proto="http", bind_tls=True)
print("MLflow Tracking UI:", ngrok_tunnel.public_url)
```

## Training the Model

1. Model is Spacy:
                Training Data Set = jsonl formatted file (90 resumes used for training)
                Test Data Set = 
2. Blank spacy model is created with ner pipeline and disabling anyother pipeline than ner and custom labels are added into the ner pipeline.

3. -- Running .py files and flowchart for it..

4. In training shuffled and batch training concept is used for getting better training of the model.

5. In testing phase each jsonl line will be taken as input as all predictions from the text are generated into the individual text files.
    
## Running MLFlow

1. File(.py) for running mlflow spacy and see the results on the frontend

