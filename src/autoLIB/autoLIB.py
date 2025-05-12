import nltk
import stanza
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag

# Initialization
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('vader_lexicon', quiet=True)
stanza.download('en', processors='tokenize,sentiment', verbose=False)


class AutoLIB:
    ABSTRACTION_SCORES = {'DAV': 1, 'IAV': 2, 'SV': 3, 'ADJ': 4, 'NN': 5}

    def __init__(self, sentiment_method: str = 'stanza'):
        self.lcm_dict = self._load_default_lcm()
        self.sentiment_method = sentiment_method.lower()
        self.stanza_nlp = stanza.Pipeline('en', processors='tokenize,sentiment', tokenize_no_ssplit=True, verbose=False)
        if self.sentiment_method == 'vader':
            self.vader_analyzer = SentimentIntensityAnalyzer()

    def _load_default_lcm(self):
        try:
            # Try package-aware resource loading (preferred)
            from importlib.resources import files
            import autoLIB.data  # assumes installed as a package
            csv_path = files(autoLIB.data).joinpath("linguistic-category-model-lcm-dictionary.csv")
            with csv_path.open('r', encoding='utf-8') as f:
                df = pd.read_csv(f)

        except (ImportError, ModuleNotFoundError, AttributeError):
            # Fallback for script-based execution (e.g., python autoLIB.py)
            import os
            current_dir = os.path.dirname(__file__)
            fallback_path = os.path.join(current_dir, "data", "linguistic-category-model-lcm-dictionary.csv")
            df = pd.read_csv(fallback_path)

        return {
            'DAV': set(df.loc[df['DAV'] == 'X', 'DicTerm'].str.lower()),
            'IAV': set(df.loc[df['IAV'] == 'X', 'DicTerm'].str.lower()),
            'SV': set(df.loc[df['SV'] == 'X', 'DicTerm'].str.lower()),
        }

    def find_relevant_sentences(self, text, keywords):
        sentences = sent_tokenize(text)
        relevant = [s for s in sentences if any(k.lower() in s.lower() for k in keywords)]
        return sentences, relevant

    def analyze_sentiment(self, sentence):
        if self.sentiment_method == 'vader':
            score = self.vader_analyzer.polarity_scores(sentence)['compound']
            valence = 'positive' if score > 0.05 else 'negative' if score < -0.05 else 'neutral'
        elif self.sentiment_method == 'stanza':
            doc = self.stanza_nlp(sentence)
            sentiment_class = doc.sentences[0].sentiment
            valence = {0: 'negative', 1: 'neutral', 2: 'positive'}[sentiment_class]
            score = {0: -0.6, 1: 0.0, 2: 0.6}[sentiment_class]  # Emulated score
        else:
            raise ValueError("Sentiment method must be 'vader' or 'stanza'")
        return valence, score

    def get_lcm_category(self, word, pos):
        word = word.lower()
        for cat, words in self.lcm_dict.items():
            if word in words:
                return cat
        if pos.startswith('JJ'):
            return 'ADJ'
        if pos.startswith('NN'):
            return 'NN'
        return None

    def abstraction_score(self, sentence):
        tokens = word_tokenize(sentence)
        pos_tags = pos_tag(tokens)
        total, count = 0, 0
        for word, pos in pos_tags:
            cat = self.get_lcm_category(word, pos)
            if cat:
                total += self.ABSTRACTION_SCORES[cat]
                count += 1
        return total / count if count else None

    def calculate_bias_index(self, results):
        desirable = [r['abstraction'] for r in results if r['valence'] == 'positive' and r['abstraction'] is not None]
        undesirable = [r['abstraction'] for r in results if r['valence'] == 'negative' and r['abstraction'] is not None]
        if not desirable and not undesirable:
            return None
        desirable_m = sum(desirable) / len(desirable) if desirable else 0
        undesirable_m = sum(undesirable) / len(undesirable) if undesirable else 0
        return desirable_m - undesirable_m

    def analyze(self, text, keywords):
        all_sentences, relevant_sentences = self.find_relevant_sentences(text, keywords)
        results = []
        total_abstraction = 0
        total_score = 0
        pos_count = neg_count = neu_count = 0

        for sentence in relevant_sentences:
            valence, score = self.analyze_sentiment(sentence)
            abs_score = self.abstraction_score(sentence)
            if abs_score is not None:
                total_abstraction += abs_score
            if valence == 'positive': pos_count += 1
            elif valence == 'negative': neg_count += 1
            else: neu_count += 1

            results.append({
                'sentence': sentence,
                'valence': valence,
                'sentiment_score': score,
                'abstraction': abs_score
            })

        avg_abstraction = total_abstraction / len(relevant_sentences) if relevant_sentences else 0
        bias_index = self.calculate_bias_index(results)

        summary = {
            'bias_index': bias_index,
            'total_sentences': len(all_sentences),
            'total_relevant_sentences': len(relevant_sentences),
            'positive_relevant_sentences': pos_count,
            'negative_relevant_sentences': neg_count,
            'neutral_relevant_sentences': neu_count,
            'overall_word_count': len(word_tokenize(text)),
            'average_relevant_sentence_abstraction': avg_abstraction
        }

        return {'overall': summary, 'sentences': results}

if __name__ == "__main__":
    analyzer = AutoLIB(sentiment_method="vader")
    text = "The protesters marched peacefully. Some protesters attacked the site. Critics hated the disruption."
    keywords = ["protest", "protesters", "pipeline"]
    result = analyzer.analyze(text, keywords)
    print(result["overall"])
    for s in result["sentences"]:
        print(s)
