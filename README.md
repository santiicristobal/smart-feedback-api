# Smart Feedback API

API REST para análisis de sentimiento de feedback de usuarios, construida con Python y FastAPI.

El sistema recibe un texto como entrada y devuelve el sentimiento detectado (positivo, neutral o negativo) junto con un puntaje numérico asociado a la predicción.

---

## Challenge Arbusta 2026

**Autor:** Santiago Gabriel Cristóbal  
**Email:** santiago.cristobal.sc@gmail.com  

---

## Descripción

Smart Feedback API permite analizar opiniones o comentarios de usuarios utilizando un modelo de Machine Learning entrenado con un dataset de feedbacks etiquetados.

La API expone un endpoint simple que acepta texto en formato JSON y responde con el resultado del análisis de sentimiento.

---

## Tecnologías utilizadas

- Python  
- FastAPI  
- Uvicorn  
- scikit-learn  
- pandas  

---

## Decisiones tomadas

- Se utilizó **FastAPI** porque permite crear APIs de forma simple, es rápido y genera automáticamente la documentación, lo cual facilita las pruebas.
- Para el análisis de texto se usó **TF-IDF**, ya que es una técnica sencilla y efectiva para convertir texto en datos numéricos que un modelo pueda entender.
- El modelo elegido fue **Logistic Regression**, por ser liviano, fácil de entrenar y suficiente para este tipo de problema.
- El puntaje (`score`) representa la probabilidad de que el texto tenga un sentimiento positivo.
- Se definieron thresholds personalizados para clasificar el resultado como **positivo**, **negativo** o **neutral**, con el objetivo de evitar clasificaciones forzadas cuando el modelo no está seguro.

---

## Estructura del proyecto

```
app/
│── main.py             # Definición de la API y endpoints
│── model.py            # Entrenamiento y predicción del modelo
│── preprocessing.py    # Limpieza y normalización de texto
data/
│── reviews.csv         # Dataset de entrenamiento
```

---

## Lógica de clasificación

El modelo devuelve una probabilidad asociada al sentimiento positivo.  
En base a ese valor, el resultado se clasifica de la siguiente manera:

- **score > 0.6** → positivo  
- **score < 0.4** → negativo  
- **caso contrario** → neutral  

El puntaje representa la confianza del modelo respecto al sentimiento positivo.

---

## Cómo ejecutar el proyecto

1. Crear y activar un entorno virtual

```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```

2. Instalar dependencias

```bash
pip install -r requirements.txt
```

3. Ejecutar el servidor

```bash
uvicorn app.main:app --reload
```

La API estará disponible en:
http://127.0.0.1:8000

La documentación interactiva se puede ver en:
http://127.0.0.1:8000/docs

## Uso de la API
Endpoint
POST /analyze

Ejemplo de request

```json
{
  "text": "¡Hola equipo! Quería compartir mi experiencia con el proceso de onboarding. La verdad es que fue mucho más fácil de lo que me imaginaba. Desde el primer día, me sentí muy bienvenido y las herramientas que proporcionaron fueron super útiles. Las sesiones de capacitación estaban bien estructuradas y los tutores eran muy pacientes; respondieron a todas mis dudas sin dudarlo. Además, la comunicación fue clara y siempre estuve al tanto de lo que seguía. Siento que tengo una buena base para empezar en este nuevo rol. Creo que hicieron un gran trabajo al diseñar todo, ¡mil gracias!"
}
```

Ejemplo de response

```json
{
  "sentiment": "positivo",
  "score": 0.9498749406711353
}
```

## Notas

> Algunas frases pueden ser clasificadas incorrectamente.
> El objetivo del proyecto es mostrar la arquitectura y el flujo de una API de ML.