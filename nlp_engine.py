import nltk
import json
import time
import re
import ollama
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


class NLPEngine:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.vectorizer = TfidfVectorizer(max_features=25, stop_words='english', ngram_range=(1, 2))
        self.count_vectorizer = CountVectorizer(stop_words='english', max_features=500)
        self.lda_model = LatentDirichletAllocation(n_components=2, random_state=42)

        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)

    def analyze_content_local(self, text):
        """
        Local analysis only using Ollama (Phi-3).
        Returns: summary (12 words max), motive, tone
        """
        if not text or len(text) < 200:
            return "Content too short for analysis.", "Informative", "Neutral"

        prompt = f"""
        Analyze this blog and return ONLY a JSON object with:
        "summary": "One sentence (MAX 40 words)",
        "motive": Choose one ("Event/News", "Product/Book", "Technical Tutorial", "Academic"),
        "tone": Choose one ("Professional", "Casual", "Neutral")

        TEXT: {text[:3500]}
        """

        try:
            time.sleep(0.5)
            local_res = ollama.chat(
                model='phi3',
                messages=[{'role': 'user', 'content': prompt + "\nReturn ONLY valid JSON."}]
            )

            raw_content = local_res['message']['content']

            match = re.search(r'\{.*\}', raw_content, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return data.get("summary"), data.get("motive"), data.get("tone")

            return "Analysis unavailable.", "Academic", "Neutral"

        except Exception as e:
            print(f"❌ Analysis failed: {str(e)[:50]}")
            return "Analysis unavailable.", "Academic", "Neutral"

    def get_comment_sentiment(self, comments):
        """Calculates average sentiment score for the blog's engagement."""
        if not comments:
            return 0.0
        return sum([self.analyzer.polarity_scores(c)['compound'] for c in comments]) / len(comments)

    def run_hybrid_lda(self, text_list):
        """Groups blogs into 2 main themes based on word frequency."""
        if len(text_list) < 3:
            return {"assignments": [0] * len(text_list), "labels": {0: "General Research"}}

        dtm = self.count_vectorizer.fit_transform(text_list)
        self.lda_model.fit(dtm)
        feature_names = self.count_vectorizer.get_feature_names_out()

        topic_labels = {}
        for idx, topic in enumerate(self.lda_model.components_):
            top_words = [feature_names[i] for i in topic.argsort()[-10:]]
            topic_labels[idx] = " ".join(top_words[-3:]).title()

        return {
            "assignments": self.lda_model.transform(dtm).argmax(axis=1),
            "labels": topic_labels
        }

    def extract_corpus_keywords(self, text_list):
        """Extracts significant keywords per blog using TF-IDF."""
        if not text_list:
            return []

        tfidf_matrix = self.vectorizer.fit_transform(text_list)
        feature_names = self.vectorizer.get_feature_names_out()

        return [
            [feature_names[idx] for idx in tfidf_matrix.getrow(i).toarray()[0].argsort()[-3:][::-1]]
            for i in range(len(text_list))
        ]