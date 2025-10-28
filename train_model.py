import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import pickle

# STEP 1: LOAD DATASET
data = pd.read_csv("spam.csv", encoding='latin-1')

# only the necessary columns
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# STEP 2: PREPROCESS and CLEANING
data['label'] = data['label'].map({'ham': 0, 'spam': 1})
data.dropna(inplace=True)

# STEP 3: SPLIT INTO TRAIN/TEST
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2, random_state=42
)

# STEP 4: FEATURE EXTRACTION
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# STEP 5: MODEL TRAINING
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# STEP 6: EVALUATION
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print("✅ Model training completed successfully!")
print("📊 Accuracy:", round(accuracy * 100, 2), "%")
print("\nDetailed Report:\n", classification_report(y_test, y_pred))

# STEP 7: SAVE MODEL AND VECTORIZER
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

print("\n💾 Model and vectorizer saved as 'model.pkl' and 'vectorizer.pkl'")
