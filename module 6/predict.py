"""
Check messages from the terminal.

  python predict.py "Congratulations! You won a free prize, claim now"
  python predict.py            (interactive mode)
"""

import sys

import joblib

model = joblib.load("model.joblib")


def check(message):
    spam_probability = model.predict_proba([message])[0][1]
    label = "SPAM" if spam_probability >= 0.5 else "HAM (normal message)"
    return label, spam_probability


if __name__ == "__main__":
    if len(sys.argv) > 1:
        messages = [" ".join(sys.argv[1:])]
    else:
        print("Type a message (or press Enter to quit).")
        messages = iter(lambda: input("> ").strip(), "")
    for msg in messages:
        label, prob = check(msg)
        print(f"{label}  (spam probability: {prob:.0%})")
