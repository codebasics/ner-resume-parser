from pathlib import Path
import spacy
from fastapi import FastAPI
from typing import List, Dict
from typing import Optional
from fastapi import APIRouter
from pydantic import BaseModel

model_dir = Path("E:\\Darsh\\DP\\Internship and Works\\AtliQ Technologies\\NER Resume Model\\my_model")
id_nlp = spacy.load(model_dir)


def get_entities(sentence) -> List[Dict]:
    doc = id_nlp(sentence)
    return [(ent.text, ent.label_) for ent in doc.ents]


class NerQuery(BaseModel):
    sentence: str

router = APIRouter(
    prefix='/nlp/id',
    tags=['nlp'],
    responses={
        404: {'description': 'Not Found'}
    }
)

@router.post('/ner')
async def api_ner(query: NerQuery):
    result = get_entities(query.sentence)
    return result


app = FastAPI()
app.include_router(router)