import numpy as np


def show_top_features_per_class(model, vectorizer, n=15):
    feature_names = np.array(vectorizer.get_feature_names_out())

    if not hasattr(model, "coef_"):
        print("This model does not expose coefficients.")
        return

    classes = model.classes_

    for class_index, class_label in enumerate(classes):
        coefficients = model.coef_[class_index]

        top_indices = coefficients.argsort()[-n:][::-1]
        top_features = feature_names[top_indices]

        print("=" * 80)
        print(f"Top words/phrases for rating {class_label}:")
        print("=" * 80)

        for word in top_features:
            print(word)

        print()