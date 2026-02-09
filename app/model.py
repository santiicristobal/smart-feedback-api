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
        df = pd.read_csv("data/reviews.csv")

        df["clean_text"] = df["message"].apply(clean_text)

        X = df["clean_text"]
        y = df["sentiment"]

        X_vec = self.vectorizer.fit_transform(X)

        self.model.fit(X_vec, y)

    def predict(self, text: str):
        # Vectorizar el texto
        text_vec = self.vectorizer.transform([text])

        # Obtener probabilidades para cada clase
        probabilities = self.model.predict_proba(text_vec)[0]

        # Obtener el índice de la clase con mayor probabilidad
        best_index = probabilities.argmax()

        # Obtener el nombre de la clase
        sentiment = self.model.classes_[best_index]

        # Obtener la confianza del modelo para esa clase
        score = probabilities[best_index]

        return {
            "sentiment": sentiment,
            "score": float(score)
    }