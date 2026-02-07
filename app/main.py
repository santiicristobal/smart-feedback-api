import pandas as pd


def load_reviews(path: str) -> pd.DataFrame:
    """
    Probando cargar las reviews.
    """
    df = pd.read_csv(path, encoding="utf-8")
    return df


if __name__ == "__main__":
    df = load_reviews("data/reviews.csv")

    print("Columnas del dataset:")
    print(df.columns)

    print("\nPrimeros 5 registros:")
    print(df.head())

    print("\nCantidad de reviews por sentimiento:")
    print(df["sentiment"].value_counts())
