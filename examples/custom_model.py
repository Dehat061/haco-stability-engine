from sklearn.linear_model import LogisticRegression

def build_model():
    return LogisticRegression(
        penalty="l2",
        solver="liblinear",
        max_iter=2000,
        random_state=0,
    )
