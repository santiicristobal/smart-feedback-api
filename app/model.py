import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from app.preprocessing import clean_text


class SentimentModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.model = LogisticRegression(max_iter=1000)
        self._train()

    def _train(self):
        # Cargar dataset
        df = pd.read_csv("data/reviews.csv")

        # Preprocesamiento
        df["clean_text"] = df["message"].apply(clean_text)

        X = df["clean_text"]
        y = df["sentiment"]

        # Vectorización
        X_vec = self.vectorizer.fit_transform(X)

        # Entrenamiento
        self.model.fit(X_vec, y)

    def predict(self, text: str):
        # Vectorizar el texto nuevo
        text_vec = self.vectorizer.transform([text])

        # Probabilidad clase positiva
        score = self.model.predict_proba(text_vec)[0][1]

        if score > 0.6:
            sentiment = "positivo"
        elif score < 0.4:
            sentiment = "negativo"
        else:
            sentiment = "neutral"

        return {
            "sentiment": sentiment,
            "score": float(score)
        }