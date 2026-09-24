# Sentiment Analysis of Internship Feedback

## Objective
Analyze intern feedback text to classify it as positive or negative sentiment,
to help identify areas where intern satisfaction can be improved.

## Dataset
60 sample feedback statements (30 positive, 30 negative), generated to reflect
common internship feedback themes — mentorship quality, communication, task
structure, and support.

## Approach
1. **Text preprocessing** — lowercased text, removed punctuation/numbers,
   normalized whitespace.
2. **Vectorization** — used TF-IDF (`TfidfVectorizer`, with English stopwords
   removed) to convert text into numeric features.
3. **Model** — Logistic Regression, a standard baseline for text classification.
4. **Train/test split** — 75/25 split, stratified to keep the positive/negative
   ratio balanced in both sets.

## Results
- **Accuracy:** 1.00
- **Precision:** 1.00
- **Recall:** 1.00
- **F1-score:** 1.00

See `confusion_matrix.png` for the visual breakdown.

## Important limitation (honest note)
The perfect score above reflects the small, templated nature of this dataset
rather than a production-ready model. When tested on feedback using vocabulary
**not** present in training (e.g. "best experience of my life" instead of
"amazing"), the model misclassified both examples. This is a well-known
small-dataset overfitting issue: the model learned the specific words used
in this dataset rather than a deep understanding of sentiment.

**To make this production-ready**, the next steps would be:
- Collect a much larger, more varied real-world feedback dataset (hundreds+ rows)
- Consider a pretrained model (e.g. a Transformer-based sentiment model) for
  better generalization to unseen vocabulary
- Cross-validation instead of a single train/test split, to get a more
  reliable performance estimate on limited data

## Files
- `feedback_data.csv` — the dataset
- `sentiment_analysis.py` — full pipeline (preprocessing → training → evaluation)
- `confusion_matrix.png` — visual evaluation output

## How to run
```
pip install pandas scikit-learn matplotlib
python3 sentiment_analysis.py
```
