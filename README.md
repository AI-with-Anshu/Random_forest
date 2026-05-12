# Day 5 — Random Forest Classifier 🌲

## What I built today
Trained a **Random Forest Classifier** on the Iris flower dataset using
custom class names — Rose, Sunflower, and Lotus — instead of the default
species names. Generated a Confusion Matrix and a Feature Importance
bar chart to understand model performance visually.

---

## What is Random Forest?
A Random Forest is a collection of many Decision Trees that work together.
Each tree is trained on a random subset of the data. When predicting,
all trees vote and the majority answer wins.

**Simple analogy:** One doctor can be wrong. 100 doctors voting together
are much harder to fool. That's Random Forest!

---

## Key Learnings
- Random Forest = ensemble of Decision Trees using majority voting
- `n_estimators=100` means 100 trees vote on every prediction
- Feature Importance shows which features actually matter
- On small datasets, single Decision Tree can sometimes match Random Forest
