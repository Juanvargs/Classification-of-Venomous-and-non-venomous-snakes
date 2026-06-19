from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.mobile_inference import predict_image_bytes


app = FastAPI(title="Snake Identifier Mobile API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    country: str = Form(""),
):
    if image.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
        raise HTTPException(status_code=400, detail="Sube una imagen JPG o PNG.")

    try:
        image_bytes = await image.read()
        return predict_image_bytes(image_bytes, country=country)
    except FileNotFoundError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"No se pudo analizar la imagen: {error}") from error
