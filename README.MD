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

```from autoLIB import AutoLIB```

## 📄 Citation

### APA 7 Format:

Collins, K. A., & Boyd, R. L. (2025). Automating the detection of linguistic intergroup bias through computerized language analysis. Journal of Language and Social Psychology, 0261927X251318887. https://doi.org/10.1177/0261927X251318887

### Bibtex

```

@article{collins_automating_2025,
	title = {Automating the detection of linguistic intergroup bias through computerized language analysis},
	issn = {0261-927X},
	url = {https://doi.org/10.1177/0261927X251318887},
	doi = {10.1177/0261927X251318887},
	abstract = {Linguistic bias is the differential use of abstraction, or other linguistic mechanisms, for the same behavior by members of different groups. Abstraction is defined by the Linguistic Category Model (LCM), which defines a continuum of words from concrete to abstract. Linguistic Intergroup Bias (LIB) characterizes the tendency for people to use abstract words for undesirable outgroup and desirable ingroup behavior and concrete words for desirable outgroup and undesirable ingroup behavior. Thus, by examining abstraction in a text, we can understand the implicit attitudes of the author. Yet, research is currently stifled by the time-consuming and resource-intensive method of manual coding. In this study, we aim to develop an automated method to code for LIB. We compiled various techniques, including forms of sentence tokenization, sentiment analysis, and abstraction coding. All methods provided scores that were a good approximation of manually coded scores, which is promising and suggests that more complex methods for LIB coding may be unnecessary. We recommend automated approaches using CoreNLP sentiment analysis and LCM Dictionary abstraction coding.},
	language = {EN},
	urldate = {2025-03-13},
	journal = {Journal of Language and Social Psychology},
	author = {Collins, Katherine A. and Boyd, Ryan L.},
	month = feb,
	year = {2025},
	note = {Publisher: SAGE Publications Inc},
	pages = {0261927X251318887},
}

```