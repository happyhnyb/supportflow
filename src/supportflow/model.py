from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .constants import INTENTS
from .privacy import normalise


@dataclass
class TicketClassifier:
    vectorizer: TfidfVectorizer
    classifier: LogisticRegression
    intents: tuple[str, ...] = INTENTS

    @classmethod
    def create(cls) -> "TicketClassifier":
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        classifier = LogisticRegression(
            C=8.0,
            max_iter=2_000,
            class_weight="balanced",
            random_state=42,
        )
        return cls(
            vectorizer=vectorizer,
            classifier=classifier,
        )

    def fit(self, messages: pd.Series, intents: pd.Series) -> "TicketClassifier":
        unknown = set(intents.unique()) - set(self.intents)
        if unknown:
            raise ValueError(f"Unsupported intents: {sorted(unknown)}")
        self.classifier.fit(self.vectorizer.fit_transform(messages.map(normalise)), intents)
        return self

    def predict_proba(self, messages: pd.Series) -> pd.DataFrame:
        clean_messages = messages.map(normalise)
        features = self.vectorizer.transform(clean_messages)
        values = self.classifier.predict_proba(features)
        table = pd.DataFrame(values, columns=self.classifier.classes_, index=messages.index)
        return table.reindex(columns=self.intents, fill_value=0.0)
