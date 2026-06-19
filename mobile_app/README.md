# App movil

MVP en Expo/React Native para enviar una foto al backend local de inferencia.

## Backend

Desde la raiz del proyecto:

```powershell
pip install -r requirements.txt
uvicorn api_mobile:app --host 0.0.0.0 --port 8000
```

## App

`API_URL` en `App.js` quedo configurado con `http://192.168.10.19:8000`.
Si cambia la red Wi-Fi, reemplaza esa IP por la nueva IP local del computador.
El backend movil usa como modelo activo `B4 EfficientNetV2B1` con politica de decision conservadora.

```powershell
cd mobile_app
npm install
npm start
```

Abre la app con Expo Go en el celular. El celular y el computador deben estar en la misma red Wi-Fi.
