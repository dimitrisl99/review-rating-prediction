import numpy as np


def explain_prediction(text, model, vectorizer, top_n=8):
    if not hasattr(model, "coef_"):
        return []

    text_vector = vectorizer.transform([text])
    feature_names = np.array(vectorizer.get_feature_names_out())

    prediction = model.predict(text_vector)[0]
    classes = list(model.classes_)

    non_zero_indices = text_vector.nonzero()[1]

    feature_scores = []

    # Binary Logistic Regression case:
    # coef_[0] usually explains the positive direction for classes_[1]
    if model.coef_.shape[0] == 1 and len(classes) == 2:
        positive_class = classes[1]
        sign = 1 if prediction == positive_class else -1
        coefficients = model.coef_[0] * sign

    else:
        class_index = classes.index(prediction)
        coefficients = model.coef_[class_index]

    for index in non_zero_indices:
        feature = feature_names[index]
        tfidf_value = text_vector[0, index]
        coefficient = coefficients[index]
        score = tfidf_value * coefficient

        feature_scores.append((feature, score))

    feature_scores = sorted(
        feature_scores,
        key=lambda item: abs(item[1]),
        reverse=True
    )

    return feature_scores[:top_n]