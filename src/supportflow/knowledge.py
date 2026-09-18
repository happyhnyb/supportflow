from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .privacy import normalise


@dataclass
class KnowledgeBase:
    articles: list[dict]
    vectorizer: TfidfVectorizer
    matrix: object

    @classmethod
    def load(cls, path: Path) -> "KnowledgeBase":
        articles = json.loads(path.read_text(encoding="utf-8"))
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        article_text = [
            f"{article['title']} {article['content']} {article['intent']}"
            for article in articles
        ]
        matrix = vectorizer.fit_transform(article_text)
        return cls(articles, vectorizer, matrix)

    def search(self, query: str, limit: int = 3, preferred_intent: str | None = None) -> list[dict]:
        query_vector = self.vectorizer.transform([normalise(query)])
        scores = (self.matrix @ query_vector.T).toarray().ravel()

        if preferred_intent:
            intent_boost = np.array(
                [0.35 if article["intent"] == preferred_intent else 0.0 for article in self.articles]
            )
            scores += intent_boost

        candidates = [
            index
            for index, article in enumerate(self.articles)
            if preferred_intent is None or article["intent"] == preferred_intent
        ]
        ranked = sorted(candidates, key=scores.__getitem__, reverse=True)[:limit]

        matches = []
        for index in ranked:
            if scores[index] <= 0:
                continue
            article = self.articles[index]
            matches.append(
                {
                    "title": article["title"],
                    "article_id": article["article_id"],
                    "snippet": article["content"],
                    "score": round(float(scores[index]), 4),
                }
            )
        return matches
