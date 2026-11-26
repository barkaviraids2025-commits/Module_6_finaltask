# 📩 SMS Spam Detector (Final AI & ML Project)

A machine learning project that reads a text message and decides whether it is **spam** or a normal message (**ham**).
Built end to end: data cleaning → exploration → model comparison and tuning → evaluation → explanation → a web app.

**Author:** Barkavi R

## Results at a glance

| Item | Value |
|---|---|
| Task | Binary text classification (spam vs ham) |
| Data | 5,572 real SMS messages, 5,169 after removing 403 duplicates |
| Final model | TF-IDF + Logistic Regression |
| Test accuracy | **97.8%** |
| Spam precision / recall / F1 | **91.5% / 90.8% / 91.2%** |
| Test set outcome (1,034 messages) | 119 spam caught, 12 missed, 11 normal messages wrongly flagged, 892 correctly kept |

> Only about 1 in 8 messages is spam, so a model that always says "ham" would score ~87% accuracy. That is why I judge the model on **precision, recall and F1**, not accuracy alone.

## The problem (plain language)

Spam filters work like a watchman who has read thousands of messages. He remembers which words usually appear in junk ("claim", "prize", "txt") and which appear in normal chats ("ok", "home", "later"). This project trains a model to learn those patterns automatically.

## Workflow

| Step | What I did |
|---|---|
| 1. Clean | Removed 403 duplicate messages. The same SMS in both train and test sets would inflate the score. |
| 2. Explore | Found the data is imbalanced and that spam messages are longer on average |
| 3. Features | **TF-IDF** on words and word pairs (stop words removed, rare words dropped) |
| 4. Models | Compared **Naive Bayes, Logistic Regression, Linear SVM**, each tuned with GridSearchCV (5-fold, optimising F1) |
| 5. Select | Chose the model using **cross-validation F1**, so the test set stayed a fair final exam |
| 6. Evaluate | Precision, recall, F1, confusion matrix, error analysis |
| 7. Explain | Charted the words that push a message towards spam or ham |
| 8. Deploy | Saved the model and built a Streamlit web app and a command-line checker |

## Model comparison

| Model | CV F1 | Test accuracy | Test precision | Test recall | Test F1 |
|---|---|---|---|---|---|
| Naive Bayes | 0.924 | 98.4% | 97.5% | 89.3% | 93.2% |
| **Logistic Regression** (chosen) | **0.930** | 97.8% | 91.5% | 90.8% | 91.2% |
| Linear SVM | 0.928 | 97.7% | 93.5% | 87.8% | 90.6% |

Honest note: Logistic Regression won on cross-validation, so I picked it without looking at test scores. Naive Bayes actually had higher test precision (fewer normal messages wrongly blocked). The three models are very close, so a precision-focused version (for example NB, or a higher decision threshold) is a good next step because blocking a real message is usually worse than letting one spam through.

## Charts

| Data exploration | What the model learned |
|---|---|
| ![Data exploration](images/eda.png) | ![Top words](images/top_words.png) |

![Confusion matrix](images/confusion_matrix.png)

## Try it

Install the libraries:

```bash
pip install -r requirements.txt
```

**Web app**

```bash
streamlit run app.py
```

**Command line**

```bash
python predict.py "Congratulations! You won a FREE prize. Claim now, reply WIN to 87121"
# SPAM  (spam probability: 100%)
```

**Retrain from scratch** (recreates `model.joblib`, `metrics.json` and the charts):

```bash
python train.py
```

**Notebook:** open `spam_detector.ipynb` in Jupyter or Google Colab for the full step-by-step walkthrough with explanations. It downloads the data automatically if `data/sms.tsv` is missing.

## Project structure

| File | Purpose |
|---|---|
| `spam_detector.ipynb` | Full walkthrough with explanations (EDA, models, evaluation, explainability) |
| `train.py` | Training script: tunes models, saves the best one and charts |
| `app.py` | Streamlit web app |
| `predict.py` | Command-line checker |
| `model.joblib` | Trained model (TF-IDF + Logistic Regression pipeline) |
| `metrics.json` | Saved results and example mistakes |
| `data/sms.tsv` | Dataset |
| `images/` | Charts used in this README |

## What I learned

- **Accuracy can mislead** on imbalanced data. Precision, recall and F1 give the real picture.
- **Data leakage is easy to cause.** Duplicates across train and test make results look better than they are.
- **Pipelines** keep preprocessing and the model together, so training and prediction use identical steps.
- **Choose with cross-validation** and use the test set only once at the end.
- **Explainability builds trust.** The top-words chart shows the model learned sensible patterns, not random noise.

## Limitations

- The data is older English SMS, so modern scams (UPI or OTP fraud, WhatsApp links, regional languages) may look different.
- The model sees only words, not the sender or the intention behind a message. Short, casual-sounding spam can slip through.
- It is a learning project and not a replacement for a real spam filter.

## Future work

- Tune the decision threshold to favour precision (fewer false alarms)
- Add features such as message length, digits and links
- Compare with a transformer model (e.g. BERT)
- Deploy the app online with Streamlit Community Cloud

## Dataset credit

SMS Spam Collection by Tiago A. Almeida and José María Gómez Hidalgo (UCI Machine Learning Repository). Used here for educational purposes.
