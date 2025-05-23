import cv2
import easyocr
import os
import sys
import contextlib
import logging
from collections import Counter, deque
import paho.mqtt.client as mqtt
from datetime import datetime
import json
from ultralytics import YOLO
import re

# ---------- CONFIGURACIÓN GENERAL ----------
skip_frames = 2
ventana_placas = deque(maxlen=45)
placas_confirmadas = set()
reader = easyocr.Reader(['en'])

# ---------- CONFIGURACIÓN MQTT ----------
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883

TOPIC_CARRO = "placas/vehiculos/adentro/carro"
TOPIC_MOTO = "placas/vehiculos/adentro/moto"

mqtt_client = mqtt.Client()
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)

# ---------- SILENCIAR SALIDA ----------
logging.getLogger('ultralytics').setLevel(logging.CRITICAL)
@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as fnull:
        old_stdout = sys.stdout
        sys.stdout = fnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

# ---------- CARGA DE MODELO (USO DE GPU) ----------
with suppress_stdout():
    modelo_yolo = YOLO('yolov8n.pt').to("cuda")
    modelo_yolo.fuse()

# ---------- CÁMARA ----------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

# ---------- VIDEO ----------
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('video_resultado.mp4', fourcc, 20.0, (640, 480))

frame_count = 0

# ---------- FUNCIONES DE VALIDACIÓN ----------
def es_placa_carro(texto):
    return re.fullmatch(r'[A-Z]{3} [0-9]{3}', texto) is not None

def es_placa_moto(texto):
    return re.fullmatch(r'[A-Z]{3} [0-9]{2}[A-Z]', texto) is not None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    detecto_algo = False
    placas_detectadas = []

    if frame_count % skip_frames == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ocr_results = reader.readtext(gray)

        for (bbox, text, conf) in ocr_results:
            text = text.strip()
            if len(text) >= 5 and conf > 0.5:
                detecto_algo = True
                placas_detectadas.append(text)
                ventana_placas.append(text)

                (tl, tr, br, bl) = bbox
                tl = tuple([int(v) for v in tl])
                br = tuple([int(v) for v in br])
                cv2.rectangle(frame, tl, br, (255, 0, 0), 2)
                cv2.putText(frame, text, (tl[0], tl[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Confirmación por repetición y validación
    if len(ventana_placas) >= 7:
        conteo = Counter(ventana_placas)
        for placa, repeticiones in conteo.items():
            if repeticiones >= 6 and placa not in placas_confirmadas:
                if es_placa_carro(placa):
                    mqtt_topic = TOPIC_CARRO
                elif es_placa_moto(placa):
                    mqtt_topic = TOPIC_MOTO
                else:
                    continue  # placa inválida, no enviar nada

                placas_confirmadas.add(placa)
                # ✅ Hora con formato completo: YYYY-MM-DD HH:MM:SS
                hora_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                fila = [{"placa": placa, "hora": hora_actual}]
                mensaje_json = json.dumps(fila)

                print(f"✅ Placa CONFIRMADA: {placa} a las {hora_actual}")
                mqtt_client.publish(mqtt_topic, mensaje_json)
                print(f"📡 Enviado MQTT a {mqtt_topic}: {mensaje_json}")

    cv2.imshow("Reconocimiento en Tiempo Real", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
out.release()
cv2.destroyAllWindows()
mqtt_client.disconnect()
