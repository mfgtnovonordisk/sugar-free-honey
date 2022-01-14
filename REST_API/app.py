from fastapi import FastAPI
import uvicorn
from tensorflow import keras
model = keras.models.load_model('/models/bee_model.h5')

app = FastAPI()

@app.get("/")
def read_root():
    return {"API status": "Bee cool!"}

@app.post("/predict/")
def predict_price(image):
    return {"prediction": model.predict(image)}