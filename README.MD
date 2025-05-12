# autoLIB

**Automated Linguistic Intergroup Bias Detection in Text**

`autoLIB` is a Python package for detecting linguistic intergroup bias (LIB) in natural language text. It combines rule-based abstraction coding (via the Linguistic Category Model) with sentence-level sentiment analysis to produce interpretable metrics of language bias. The package is designed to support social psychology, media studies, and computational linguistics research.

---

## ✨ Features

- 📄 **Sentence-level valence classification** using VADER or Stanza sentiment models.
- 🔤 **Word-level abstraction scoring** based on the Linguistic Category Model (LCM).
- 🧠 **Bias index computation**: compares abstraction in desirable vs. undesirable descriptions.
- 🔍 **Keyword-driven sentence filtering** to isolate relevant parts of text.
- 📊 Outputs sentence-level results and overall summary statistics.
- 🧪 Designed with transparency, replicability, and academic applications in mind.

---

## 📦 Installation

The simplest way to install this package is via the pypi build:
```pip install autoLIB```

Alternatively, you can clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/autoLIB.git
cd autoLIB
pip install -r requirements.txt
