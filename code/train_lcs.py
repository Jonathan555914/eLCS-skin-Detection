"""Train local scikit-eLCS on extracted feature CSV files.

This script intentionally imports from ``code/scikit-eLCS`` instead of a
pip-installed package so algorithm edits remain local to this project.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_ELCS_SRC = PROJECT_ROOT / "code" / "scikit-eLCS"
sys.path.insert(0, str(LOCAL_ELCS_SRC))

from skeLCS import eLCS  # noqa: E402


def remove_outlier_rows(
    df: pd.DataFrame,
    class_column: str,
    z_threshold: float = 4.0,
    absolute_threshold: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove whole records containing extreme feature values.

    The z-score rule catches values far away from the rest of a feature column.
    ``absolute_threshold`` is optional for obvious errors such as 10000 when the
    normal range is roughly 2 to 700.
    """
    feature_df = df.drop(columns=[class_column])
    numeric_features = feature_df.apply(pd.to_numeric, errors="coerce")
    std = numeric_features.std(axis=0).replace(0, np.nan)
    z_scores = ((numeric_features - numeric_features.mean(axis=0)) / std).abs()
    outlier_mask = z_scores.gt(z_threshold).any(axis=1)

    if absolute_threshold is not None:
        outlier_mask = outlier_mask | numeric_features.abs().gt(absolute_threshold).any(axis=1)

    removed = df.loc[outlier_mask].copy()
    cleaned = df.loc[~outlier_mask].copy()
    return cleaned, removed


def verify_minmax_scaled(name: str, values: np.ndarray, tolerance: float = 1e-8) -> None:
    """Verify training scaling and report test-set extrapolation.

    MinMaxScaler guarantees [0, 1] only for the data used by ``fit``. Test
    values outside the training minimum/maximum legitimately transform below
    zero or above one and must not cause data leakage by refitting the scaler.
    """
    col_min = np.nanmin(values, axis=0)
    col_max = np.nanmax(values, axis=0)
    global_min = float(np.nanmin(col_min))
    global_max = float(np.nanmax(col_max))
    print(f"{name} normalized range: min={global_min:.6f}, max={global_max:.6f}")

    if name == "train":
        if global_min < -tolerance or global_max > 1.0 + tolerance:
            raise ValueError("Training data has values outside [0, 1]. Check preprocessing.")

        non_constant = ~np.isclose(col_min, col_max)
        bad_min = np.where((np.abs(col_min) > tolerance) & non_constant)[0]
        bad_max = np.where((np.abs(col_max - 1.0) > tolerance) & non_constant)[0]
        if len(bad_min) or len(bad_max):
            raise ValueError("Training columns are not scaled to min=0 and max=1.")
    else:
        outside_count = int(np.count_nonzero((values < -tolerance) | (values > 1.0 + tolerance)))
        if outside_count:
            print(
                f"{name} contains {outside_count} values outside [0, 1]; "
                "this is expected when test values exceed the training range."
            )


def load_feature_table(path: Path, class_column: str) -> pd.DataFrame:
    """Load CSV or TSV feature table."""
    separator = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    df = pd.read_csv(path, sep=separator)
    if class_column not in df.columns:
        raise ValueError(f"Class column '{class_column}' not found in {path}")
    return df


def decode_label(label_encoder: LabelEncoder, value):
    """Convert eLCS numeric class labels back to the original class names."""
    return label_encoder.inverse_transform([int(float(value))])[0]


def inverse_minmax_value(scaler: MinMaxScaler, feature_index: int, value: float) -> float:
    """Convert one normalized feature value back to the original feature scale."""
    return float(value * (scaler.data_max_[feature_index] - scaler.data_min_[feature_index]) + scaler.data_min_[feature_index])


def format_classifier_condition(model: eLCS, classifier, feature_names: list[str], scaler: MinMaxScaler) -> tuple[str, str]:
    """Return human-readable rule conditions in normalized and original scales."""
    normalized_parts = []
    original_parts = []

    for attribute_index, condition in zip(classifier.specifiedAttList, classifier.condition):
        feature_name = feature_names[attribute_index]
        is_continuous = model.env.formatData.attributeInfoType[attribute_index]

        if is_continuous:
            norm_low = float(condition[0])
            norm_high = float(condition[1])
            original_low = inverse_minmax_value(scaler, attribute_index, norm_low)
            original_high = inverse_minmax_value(scaler, attribute_index, norm_high)
            normalized_parts.append(f"{feature_name} in [{norm_low:.6g}, {norm_high:.6g}]")
            original_parts.append(f"{feature_name} in [{original_low:.6g}, {original_high:.6g}]")
        else:
            normalized_parts.append(f"{feature_name} == {condition}")
            original_parts.append(f"{feature_name} == {condition}")

    return " AND ".join(normalized_parts), " AND ".join(original_parts)


def export_rule_population(
    model: eLCS,
    feature_names: list[str],
    scaler: MinMaxScaler,
    label_encoder: LabelEncoder,
    output_file: Path,
) -> None:
    """Export the learned eLCS rules as readable IF-THEN rows."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for rule_id, classifier in enumerate(model.population.popSet):
        normalized_condition, original_condition = format_classifier_condition(model, classifier, feature_names, scaler)
        predicted_class = decode_label(label_encoder, classifier.phenotype)
        rows.append(
            {
                "rule_id": rule_id,
                "if_normalized": normalized_condition,
                "if_original_scale": original_condition,
                "then_class": predicted_class,
                "fitness": classifier.fitness,
                "accuracy": classifier.accuracy,
                "numerosity": classifier.numerosity,
                "match_count": classifier.matchCount,
                "correct_count": classifier.correctCount,
                "specificity": len(classifier.specifiedAttList) / len(feature_names),
            }
        )

    rule_df = pd.DataFrame(rows)
    if not rule_df.empty:
        rule_df = rule_df.sort_values(["accuracy", "fitness", "numerosity"], ascending=False)
    rule_df.to_csv(output_file, index=False)
    print(f"Exported readable eLCS rules: {output_file}")


def train_from_feature_file(
    feature_file: Path,
    class_column: str,
    learning_iterations: int,
    test_size: float,
    random_state: int,
    z_threshold: float,
    absolute_threshold: float | None,
    rules_output: Path,
) -> None:
    df = load_feature_table(feature_file, class_column)
    print(f"Loaded rows: {len(df)} from {feature_file}")

    cleaned, removed = remove_outlier_rows(df, class_column, z_threshold, absolute_threshold)
    print(f"Removed outlier rows: {len(removed)}")
    print(f"Rows after outlier removal: {len(cleaned)}")

    feature_names = list(cleaned.drop(columns=[class_column]).columns)
    X = cleaned.drop(columns=[class_column]).apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(cleaned[class_column].astype(str).to_numpy()).astype(float)
    print("Class label mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    verify_minmax_scaled("train", X_train)
    verify_minmax_scaled("test", X_test)

    model = eLCS(
        learning_iterations=learning_iterations,
        track_accuracy_while_fit=True,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    y_test_labels = label_encoder.inverse_transform(y_test.astype(int))
    prediction_labels = label_encoder.inverse_transform(predictions.astype(int))

    print("Balanced accuracy:", balanced_accuracy_score(y_test_labels, prediction_labels))
    print(classification_report(y_test_labels, prediction_labels))
    export_rule_population(model, feature_names, scaler, label_encoder, rules_output)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train local scikit-eLCS on one feature file.")
    parser.add_argument("feature_file", type=Path)
    parser.add_argument("--class-column", default="Class")
    parser.add_argument("--learning-iterations", type=int, default=5000)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--z-threshold", type=float, default=4.0)
    parser.add_argument("--absolute-threshold", type=float, default=None)
    parser.add_argument("--rules-output", type=Path, default=PROJECT_ROOT / "metadata" / "results" / "eLCS_rules.csv")
    args = parser.parse_args()

    train_from_feature_file(
        args.feature_file,
        args.class_column,
        args.learning_iterations,
        args.test_size,
        args.random_state,
        args.z_threshold,
        args.absolute_threshold,
        args.rules_output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
