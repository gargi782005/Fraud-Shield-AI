from sklearn.ensemble import RandomForestClassifier


def test_random_forest_model_can_be_created():
    model = RandomForestClassifier(
        n_estimators=5,
        max_depth=3,
        random_state=42
    )

    assert model is not None


def test_random_forest_model_can_train():
    X = [
        [0.0, 1.0],
        [1.0, 0.0],
        [0.1, 1.1],
        [1.1, 0.1]
    ]

    y = [0, 1, 0, 1]

    model = RandomForestClassifier(
        n_estimators=5,
        max_depth=3,
        random_state=42
    )

    model.fit(X, y)

    prediction = model.predict([[0.2, 1.0]])

    assert prediction[0] in [0, 1]