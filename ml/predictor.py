from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ml.feature_engineering import FEATURE_COLUMNS, build_training_dataset, latest_feature_row


@dataclass(frozen=True)
class PredictionResult:
    probability_up: float | None
    confidence: str
    feature_importance: pd.DataFrame
    explanation: list[str]
    train_rows: int
    validation_accuracy: float | None
    available: bool


class StockMLPredictor:
    def __init__(self, min_rows: int = 80, random_state: int = 42) -> None:
        self.min_rows = min_rows
        self.random_state = random_state

    def predict_next_period(self, frame: pd.DataFrame) -> PredictionResult:
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score
        except Exception:
            return _not_available("scikit-learn is not installed. Install requirements.txt to enable ML predictions.")

        x, y = build_training_dataset(frame)
        latest = latest_feature_row(frame)
        if len(x) < self.min_rows or latest.empty:
            return _not_available("Not enough clean historical rows to train a reliable per-stock model.")
        if y.nunique() < 2:
            return _not_available("Historical target has only one class, so the model cannot learn both outcomes.")

        split = max(int(len(x) * 0.8), 1)
        x_train, x_valid = x.iloc[:split], x.iloc[split:]
        y_train, y_valid = y.iloc[:split], y.iloc[split:]
        model = RandomForestClassifier(
            n_estimators=250,
            max_depth=6,
            min_samples_leaf=5,
            random_state=self.random_state,
            class_weight="balanced_subsample",
        )
        model.fit(x_train, y_train)
        validation_accuracy = None
        if len(x_valid) >= 10 and y_valid.nunique() >= 2:
            validation_accuracy = float(accuracy_score(y_valid, model.predict(x_valid)))

        probability = float(model.predict_proba(latest[FEATURE_COLUMNS])[0][1])
        importances = pd.DataFrame(
            {
                "Feature": FEATURE_COLUMNS,
                "Importance": model.feature_importances_,
            }
        ).sort_values("Importance", ascending=False)

        confidence = _confidence_label(probability, len(x), validation_accuracy)
        top_features = importances.head(3)
        explanation = [
            f"Model was trained on {len(x)} clean historical observations for this stock.",
            f"Estimated next-period up probability is {probability * 100:.1f}%.",
            "Top drivers: " + ", ".join(f"{row.Feature} ({row.Importance:.2f})" for row in top_features.itertuples()),
        ]
        if validation_accuracy is not None:
            explanation.append(f"Recent validation accuracy is {validation_accuracy * 100:.1f}%; treat this as diagnostic, not a guarantee.")
        else:
            explanation.append("Validation sample is small, so confidence is capped.")

        return PredictionResult(
            probability_up=round(probability * 100, 2),
            confidence=confidence,
            feature_importance=importances.reset_index(drop=True),
            explanation=explanation,
            train_rows=len(x),
            validation_accuracy=None if validation_accuracy is None else round(validation_accuracy * 100, 2),
            available=True,
        )


def _not_available(message: str) -> PredictionResult:
    return PredictionResult(
        probability_up=None,
        confidence="Low",
        feature_importance=pd.DataFrame(columns=["Feature", "Importance"]),
        explanation=[message],
        train_rows=0,
        validation_accuracy=None,
        available=False,
    )


def _confidence_label(probability: float, rows: int, validation_accuracy: float | None) -> str:
    distance = abs(probability - 0.5)
    if rows >= 220 and distance >= 0.18 and (validation_accuracy is None or validation_accuracy >= 0.55):
        return "High"
    if rows >= 120 and distance >= 0.10:
        return "Medium"
    return "Low"
