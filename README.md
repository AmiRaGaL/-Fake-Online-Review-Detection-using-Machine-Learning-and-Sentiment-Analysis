# Fake Online Review Detection using Machine Learning 

## Introduction and Motivation
The objective of this project is to detect fake online reviews using machine learning techniques. Fake reviews are artificially created to mislead customers or inflate product ratings, posing a risk to both businesses and consumers. Our goal is to build a model that accurately identifies fraudulent reviews, enhancing trust in online review systems.

### Problem Definition
Detecting fake reviews is essential for maintaining credibility on online platforms. This is achieved by training a model on a labeled dataset of genuine and fraudulent reviews, analyzing various features like text content, sentiment, and writing style.

### Importance of the Project
Detecting fake reviews is critical for:
- **Businesses:** Protects their reputation and values genuine feedback.
- **Consumers:** Helps them make informed purchasing decisions.
- **Online Platforms:** Enhances transparency, increasing user trust.

## Beneficiaries of the Solution
- **Businesses:** Improve online reputation and attract genuine customers.
- **Consumers:** Access to reliable reviews leads to better decision-making.
- **Online Platforms:** Builds user trust and engagement.

## Baseline Model
Our baseline model employs three machine learning classifiers:
1. **Multinomial Naive Bayes (MNB)**
2. **Support Vector Machine (SVM)**
3. **Logistic Regression (LR)**

Using two vectorization techniques—**CountVectorizer** and **TF-IDF Vectorizer**—we trained these models on labeled review data. Among the classifiers, Logistic Regression achieved the highest performance with:
- **Accuracy:** 85%
- **Recall:** 92%

This model can detect fraudulent reviews and assign a probability score to user-submitted reviews.

## Dataset
We used the **Amazon Customer Reviews Dataset**, a comprehensive collection of over 100 million reviews contributed by Amazon customers since 1995. The dataset provides rich information for NLP, IR, and ML research, containing fields such as:
- `marketplace`: Country code for the review's origin.
- `review_id`: Unique identifier for each review.
- `product_id`: ID of the reviewed product.
- `product_category`: Broad product category.
- `star_rating`: 1-5 star rating given by the reviewer.
- `helpful_votes`: Number of helpful votes for the review.
- `total_votes`: Total number of votes.
- Additional attributes like `vine`, `verified_purchase`, `review_headline`, and `review_body`.

## Evaluation Metrics
To evaluate our model, we utilized the following metrics:
- **Accuracy:** Overall correctness of the model.
- **Precision:** Ability to correctly identify fraudulent reviews.
- **Recall:** Ability to capture all fraudulent reviews.
- **F1 Score:** Harmonic mean of precision and recall, balancing false positives and false negatives.

These metrics are computed for models using both CountVectorizer and TF-IDF Vectorizer.

## Evaluation Method
Detailed evaluation methods and metrics results are recorded for each classifier and vectorization approach, assessing model accuracy and reliability in detecting fraudulent reviews.

---

This project demonstrates the viability of machine learning approaches in fraud detection, helping to maintain transparency and trust on e-commerce platforms.
"# -Fake-Online-Review-Detection-using-Machine-Learning-and-Sentiment-Analysis" 
