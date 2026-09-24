import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)

# ============================================================
# 1. LOAD DATA
# ============================================================
df = pd.read_csv('feedback_data.csv')
print("Dataset preview:")
print(df.head())
print(f"\nTotal rows: {len(df)}")
print(df['sentiment'].value_counts())

# ============================================================
# 2. TEXT PREPROCESSING
# ============================================================
# Clean text: lowercase, remove punctuation/numbers, extra spaces.
# TF-IDF handles stopwords for us later (via stop_words='english').
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)   # remove punctuation/numbers
    text = re.sub(r'\s+', ' ', text).strip()  # collapse extra spaces
    return text

df['clean_text'] = df['feedback_text'].apply(clean_text)
print("\nExample before/after cleaning:")
print("Before:", df['feedback_text'].iloc[0])
print("After: ", df['clean_text'].iloc[0])

# ============================================================
# 3. FEATURES (X) AND TARGET (y)
# ============================================================
X = df['clean_text']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
    # stratify=y keeps the positive/negative ratio balanced in both splits
)
print(f"\nTraining rows: {len(X_train)} | Testing rows: {len(X_test)}")

# ============================================================
# 4. TF-IDF VECTORIZATION (text -> numbers)
# ============================================================
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)  # learn vocab + transform train
X_test_vec = vectorizer.transform(X_test)        # transform test using SAME vocab

# ============================================================
# 5. TRAIN LOGISTIC REGRESSION MODEL
# ============================================================
model = LogisticRegression(random_state=42)
model.fit(X_train_vec, y_train)

# ============================================================
# 6. PREDICT AND EVALUATE
# ============================================================
predictions = model.predict(X_test_vec)

print("\nActual:    ", list(y_test))
print("Predicted: ", list(predictions))

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, pos_label='positive')
recall = recall_score(y_test, predictions, pos_label='positive')
f1 = f1_score(y_test, predictions, pos_label='positive')

print(f"\nAccuracy:  {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")
print(f"F1-score:  {f1:.2f}")

print("\nFull classification report:")
print(classification_report(y_test, predictions))

# ============================================================
# 7. CONFUSION MATRIX (visual)
# ============================================================
cm = confusion_matrix(y_test, predictions, labels=['positive', 'negative'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['positive', 'negative'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix - Intern Feedback Sentiment')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
print("\nSaved confusion_matrix.png")

# ============================================================
# 8. TEST ON NEW / UNSEEN FEEDBACK (to demo generalization)
# ============================================================
# NOTE: with only 60 rows, the model mainly learns the exact vocabulary
# used in this dataset (amazing, excellent, poor, frustrating, etc.).
# Feedback using DIFFERENT words than the training vocabulary may be
# misclassified - this is a known limitation of small text datasets,
# not a bug. It's worth stating this explicitly in your report.
new_feedback = [
    "The internship was amazing and the mentors were very supportive",   # uses training vocab
    "The tasks were poor and the team felt disorganized",                # uses training vocab
    "This was one of the best experiences of my life, learned so much",  # NEW vocab (out of distribution)
    "Terrible experience, nobody replied to my messages"                 # NEW vocab (out of distribution)
]
new_clean = [clean_text(t) for t in new_feedback]
new_vec = vectorizer.transform(new_clean)
new_predictions = model.predict(new_vec)

print("\nNew feedback predictions:")
for text, pred in zip(new_feedback, new_predictions):
    print(f"  '{text}' -> {pred}")
print("\n(Note: the last two examples use vocabulary NOT seen in training,")
print("so they demonstrate the model's generalization limits with this small dataset.)")
