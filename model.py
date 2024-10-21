import numpy as np
import joblib
import data
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Split the data
X_train, X_test, y_train, y_test = train_test_split(data.reviews, data.labels, test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)

# Train the model
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

# Save the model and vectorizer
joblib.dump(model, 'hotel_review_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

# Evaluate the model
X_test_vectorized = vectorizer.transform(X_test)
predictions = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, predictions)
print(f"Model accuracy: {accuracy * 100:.2f}%")
