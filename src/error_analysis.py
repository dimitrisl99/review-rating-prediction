def show_misclassified_examples(X_test, y_test, y_pred, n=10):
    errors_found = 0

    for review, true_label, pred_label in zip(X_test, y_test, y_pred):
        if true_label != pred_label:
            print("=" * 80)
            print(f"TRUE LABEL: {true_label}")
            print(f"PREDICTED : {pred_label}")
            print("REVIEW:")
            print(review)
            print("=" * 80)
            print()

            errors_found += 1

            if errors_found >= n:
                break