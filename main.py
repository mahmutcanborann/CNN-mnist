import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import uvicorn
from fastapi import FastAPI, File, UploadFile
import io
# pip install uvicorn fastapi tensorflow python-multipart pillow

app = FastAPI()

model = load_model('mnist.h5')

def preprocess_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert('L')
    img = img.resize((28, 28))
    demo = np.array(img)
    demo = demo.astype("float32") / 255
    demo = demo.reshape(-1, 28, 28, 1)
    return demo

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_bytes = await file.read()
    processed_image = preprocess_image(file_bytes)
    sonuc = model.predict(processed_image)
    sonuc = np.argmax(sonuc)
    return {"sonuc": int(sonuc)}


#tahmin=preprocess_image('8.png')
#sonuc=model.predict(tahmin)
#sonuc=np.argmax(sonuc)
#print(sonuc)