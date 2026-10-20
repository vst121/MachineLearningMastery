# Classical NLP Models and Tasks

## 🎯 Learning Objectives

- Build text classification pipelines using classical machine learning algorithms.
- Understand Multinomial Naive Bayes and Logistic Regression.
- Compare Bag of Words and TF-IDF representations.
- Apply correct training and test separation to avoid data leakage.
- Evaluate models using multiple classification metrics.
- Interpret feature weights, class probabilities, and prediction errors.
- Understand the strengths and limitations of classical NLP approaches.

## 🧠 Core Concepts

### 1. Text Classification

Text classification assigns a predefined category to a document.

Given a numerical document representation, a classifier predicts its class:

$$
\hat{y}=f(\mathbf{x})
$$

Where:

- \(\mathbf{x}\) is the numerical feature vector.
- \(f\) is the trained classification model.
- \(\hat{y}\) is the predicted class.

A typical pipeline consists of text vectorization, model training, prediction, and evaluation.

### 2. Bag of Words

Bag of Words represents documents using term counts.

$$
X_{d,w}=\text{count}(w,d)
$$

Here, \(X\_{d,w}\) is the number of occurrences of word \(w\) in document \(d\).

Binary occurrence features instead record whether a word appears:

$$
X_{d,w}=
\begin{cases}
1 & \text{if word }w\text{ occurs in document }d\\
0 & \text{otherwise}
\end{cases}
$$

Both representations produce sparse document-term matrices suitable for classical machine learning.

### 3. TF-IDF

TF-IDF assigns greater weight to terms that are informative within a document and less weight to terms appearing across many documents.

$$
\text{TFIDF}(w,d) = \text{TF}(w,d)
\left(
\log\frac{1+D}{1+\text{DF}(w)}+1
\right)
$$

Where \(D\) is the number of documents and \(\text{DF}(w)\) is the number of documents containing the term.

The exact weighting depends on the vectorizer configuration. Scikit-learn's default TF-IDF vectorizer also applies L2 normalization to nonzero document vectors.

### 4. Multinomial Naive Bayes

Naive Bayes uses Bayes' theorem to estimate the probability of a class given document features.

$$
P(c\mid\mathbf{x})
\propto
P(c)\prod_{j=1}^{V}P(w_j\mid c)^{x_j}
$$

The conditional independence assumption makes the model computationally efficient for text classification.

Using logarithms, the class score becomes:

$$
\log P(c)+
\sum_{j=1}^{V}x_j\log P(w_j\mid c)
$$

Laplace smoothing prevents unseen features from receiving zero probability:

$$
P(w\mid c)=
\frac{N_{w,c}+\alpha}
{N_c+\alpha V}
$$

Here, \(N\_{w,c}\) is the count of a term in class \(c\), \(N_c\) is the total term count for that class, \(V\) is the vocabulary size, and \(\alpha\) is the smoothing parameter.

Multinomial Naive Bayes is particularly useful as a lightweight baseline for count-based text classification. It can also work with nonnegative TF-IDF features.

### 5. Logistic Regression

Logistic Regression estimates class probabilities from a weighted combination of input features.

For binary classification:

$$
P(y=1\mid\mathbf{x}) = \frac{1}{1+e^{-(\mathbf{w}^{T}\mathbf{x}+b)}}
$$

The decision score is:

$$
z=\mathbf{w}^{T}\mathbf{x}+b
$$

A positive score favors the class associated with the positive coefficient direction; a negative score favors the other class.

The model learns feature weights that help distinguish classes. Regularization controls model complexity.

$$
J(\mathbf{w},b) = -\frac{1}{N}\sum_{i=1}^{N}
\left[
y_i\log p_i+(1-y_i)\log(1-p_i)
\right]
+
\frac{1}{2C}\|\mathbf{w}\|_2^2
$$

Here, \(C\) controls the strength of L2 regularization. Smaller values impose stronger regularization. This expression illustrates the binary classification objective; implementation details depend on the solver and configuration.

### 6. Training and Data Leakage

The training set is used to learn model parameters. The test set estimates performance on unseen examples.

Learned preprocessing steps, such as vocabulary construction and IDF calculation, must use training data only.

A scikit-learn `Pipeline` keeps the vectorizer and classifier together, ensuring that vectorization is fitted on training data and applied consistently to test data.

For reliable model selection, use cross-validation on training data and reserve the test set for final evaluation.

### 7. Classification Metrics

Accuracy measures the proportion of correct predictions:

$$
\text{Accuracy} = \frac{TP+TN}{TP+TN+FP+FN}
$$

Precision measures the reliability of positive predictions:

$$
\text{Precision}=\frac{TP}{TP+FP}
$$

Recall measures the proportion of actual positive examples detected:

$$
\text{Recall}=\frac{TP}{TP+FN}
$$

F1-score balances precision and recall:

$$
F_1=
2\frac{\text{Precision}\text{Recall}}
{\text{Precision}+\text{Recall}}
$$

Macro F1 calculates F1 independently for each class and then averages the results:

$$
F_{1,\text{macro}} = \frac{1}{C}\sum_{c=1}^{C}F_{1,c}
$$

Macro averaging gives each class equal weight, making it useful when class performance must be compared independently of class frequency.

### 8. Model Comparison

Different feature representations and algorithms can produce different results.

For a chosen evaluation metric \(M\), the best configuration among the evaluated candidates is:

$$
m^*=\arg\max_{m\in\mathcal{M}}M(m)
$$

The choice of metric should reflect the task's priorities. A single test split, especially a small one, cannot establish that a model is generally superior.

### 9. Error Analysis

A confusion matrix reveals how often each actual class is predicted as each possible class.

- **False positive:** an example is incorrectly assigned to a class.
- **False negative:** an example belonging to a class is assigned elsewhere.
- **Ambiguous text:** a document contains features associated with multiple classes.
- **Unseen vocabulary:** important terms are missing from the learned vocabulary.

For a TF-IDF vectorizer, a document containing no known terms becomes a zero vector:

$$
\mathbf{x}=\mathbf{0}
$$

For Logistic Regression, the prediction for such a document is determined by the model's intercept rather than document-specific word features.

Predicted probabilities also require care. A high maximum probability does not guarantee correctness or good probability calibration.

## 🧪 Implementation and Experiments

| File                                         | Experiment                                                                                                                                                                                  |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `01_naive_bayes_text_classification.py`      | Build a Bag of Words and Multinomial Naive Bayes pipeline. Evaluate classification metrics, inspect class probabilities and word likelihoods, and compare count-based with binary features. |
| `02_tfidf_logistic_regression.py`            | Train Logistic Regression using TF-IDF features. Inspect class probabilities, decision scores, influential feature weights, and behavior on unfamiliar vocabulary.                          |
| `03_text_classification_model_comparison.py` | Compare Bag of Words and TF-IDF with Naive Bayes and Logistic Regression using the same stratified train-test split and multiple metrics.                                                   |
| `04_text_classification_error_analysis.py`   | Inspect false positives, false negatives, class-specific metrics, ambiguous texts, and predictions for documents with no known features.                                                    |

### Tools and Libraries

- **scikit-learn:** text vectorization, classification pipelines, train-test splitting, and evaluation.
- **NumPy:** numerical operations and feature analysis.
- **pandas:** structured comparison of model results and prediction errors.

All experiments use a small, self-contained sports-versus-technology dataset. No external corpus download is required.

## ✅ Expected Outcome

After completing this module, you should be able to:

- Build classical text classification pipelines from raw documents.
- Explain the assumptions behind Multinomial Naive Bayes.
- Train and interpret Logistic Regression with sparse text features.
- Compare Bag of Words and TF-IDF representations.
- Prevent data leakage through proper pipeline design.
- Evaluate classifiers using accuracy, precision, recall, F1-score, and confusion matrices.
- Compare model-feature combinations systematically.
- Diagnose errors caused by ambiguity, limited vocabulary, and insufficient training data.
- Explain why classical NLP models remain useful as efficient baselines.

**Important:** the dataset is intentionally small for learning. Its test set contains only six documents, so reported scores are unstable and should not be interpreted as production performance. Use larger representative datasets, cross-validation, and a held-out test set for meaningful model selection.
