# 📚 Model Selection — Further Reading

A curated set of references that go deeper into the choices we make in [`2_model_selection_example.ipynb`](../2.%20Training/2_model_selection_example.ipynb): which algorithm to pick, why tree-based models dominate tabular data, and how the popular gradient-boosting libraries differ.

---

## 🗺️ Picking an algorithm — visual cheat sheets

Quick decision aids for choosing a starting model based on your problem and data shape.

- **[scikit-learn estimator map](https://scikit-learn.org/stable/machine_learning_map.html)**
  The classic interactive flowchart: answer a few yes/no questions about your dataset and it points you to a sklearn estimator. The single most useful 5-minute reference in ML.

- **[Azure ML algorithm cheat sheet](https://learn.microsoft.com/en-us/azure/machine-learning/algorithm-cheat-sheet?view=azureml-api-1)**
  Microsoft's PDF cheat sheet — same flowchart idea, broader algorithm coverage (deep learning + classical ML), printable.

- **[DataCamp ML cheat sheet](https://www.datacamp.com/cheat-sheet/machine-learning-cheat-sheet)**
  Compact one-pager mapping problem types (classification, regression, clustering, …) to representative algorithms, with strengths/weaknesses of each.

---

## 📖 Foundations and concepts

When you want to understand *why* a model works, not just which one to call.

- **[An Introduction to Statistical Learning (ISLR)](https://www.statlearning.com/)**
  The free, gold-standard introductory textbook — Python and R editions both downloadable. Chapters 4 (classification), 8 (trees), and 10 (deep learning) are the most relevant for this course.

- **[Google ML: Decision Forests](https://developers.google.com/machine-learning/decision-forests)**
  Google's own developer guide to decision trees, random forests, and gradient-boosted trees. Concise, visual, with interactive examples — pair it with our CatBoost / XGBoost notebook.

- **[ML algorithms overview — Elite Data Science](https://elitedatascience.com/machine-learning-algorithms)**
  Practitioner-oriented walkthrough of the main algorithm families with plain-English pros/cons. Good "what's out there" sweep before diving into one.

---

## 🌲 Tree-based models in depth

For the gradient-boosting libraries we actually use in Module 2.

- **["Why do tree-based models still outperform deep learning on typical tabular data?"](https://arxiv.org/abs/2207.08815)** *(Grinsztajn et al., 2022)*
  Influential paper showing that on the kind of structured, mid-sized tabular data most companies have, gradient-boosted trees still beat carefully tuned neural nets. Useful ammunition the next time someone asks "shouldn't we use deep learning?"

- **[CatBoost vs. LightGBM vs. XGBoost](https://towardsdatascience.com/catboost-vs-lightgbm-vs-xgboost-c80f40662924/)**
  Side-by-side comparison of the three industry-standard gradient boosting libraries — training speed, categorical handling, GPU support, and accuracy benchmarks.
