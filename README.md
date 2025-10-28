📧 EMAIL SPAM DETECTION USING MACHINE LEARNING

Project Overview
----------------
This is an end-to-end Machine Learning project that detects whether an email message is Spam or Not Spam.
It uses a Naive Bayes classifier trained on a spam dataset and provides a user-friendly interface built using Streamlit.

Features
--------
- Clean and preprocess text data
- Train a machine learning model (Multinomial Naive Bayes) 
- Evaluate accuracy and performance
- Save trained model and vectorizer for reuse
- Build an interactive web app using Streamlit

Technologies Used
-----------------
- Python 3
- Pandas
- Scikit-learn
- TfidfVectorizer
- Multinomial Naive Bayes
- Streamlit

Project Structure
-----------------
email_spam_detection/
│
├── spam.csv              → Dataset
├── train_model.py        → Backend: trains and saves model
├── model.pkl             → Saved ML model
├── vectorizer.pkl        → Saved text vectorizer
├── app.py                → Streamlit frontend app
└── README.docx           → Project documentation

Dataset
-------
Dataset used: Spam.csv (from Kaggle)
- Columns: v1 (label), v2 (message)
- Labels: ham → 0 (Not Spam), spam → 1 (Spam)

How to Run the Project
----------------------
1️⃣ Install Dependencies
    pip install pandas scikit-learn streamlit

2️⃣ Train the Model
    python train_model.py

   This will train the model and create:
   - model.pkl
   - vectorizer.pkl

3️⃣ Run the Streamlit App
    streamlit run app.py

   Open the local link shown in the terminal (usually http://localhost:8501).

4️⃣ Use the App
   - Enter or paste an email message
   - Click on Predict
   - The app will show whether it’s Spam or Not Spam

Example Output
--------------
Input:
  Congratulations! You have won a $1000 Walmart gift card. Click here to claim now!

Output:
  🚨 This is a SPAM message!

Future Improvements
-------------------
- Deploy on Streamlit Cloud or Hugging Face Spaces
- Add more advanced preprocessing (stemming, lemmatization)
- Try deep learning models (LSTM, BERT)

Author
------
Your Name
OM PRAKASH JENA (AI-ML)
Email: omprakashjena361@gmail.com
GitHub: https://github.com/om-jena27
LinkedIn: https://linkedin.com/in/omjena
