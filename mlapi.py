#bring in light
from fastapi import FastAPI
import numpy as np
from pydantic import BaseModel
import pickle

app = FastAPI()
pkl_filename = "text_categorization_model.pkl"
with open(pkl_filename, 'rb') as f_in:
        model = pickle.load(f_in)
with open('text_categorization_model.pkl', 'rb') as file:
    model = pickle.load(file)
with open('tokenizer.pkl', 'rb') as t:
    tokenizer = pickle.load(t)
with open('encoder.pkl', 'rb') as e:
    encoder = pickle.load(e)

class Item(BaseModel):
    text: str




@app.post("/predict/")
async def predict(item: Item):
    # Preprocess the input
    x_new = tokenizer.texts_to_matrix([item.text])  # Assuming this method exists for your tokenizer

    # Predict
    predictions = model.predict(x_new)
    predicted_label = encoder.inverse_transform([np.argmax(predictions)])[0]  # Decode the prediction

    return {"prediction": predicted_label}