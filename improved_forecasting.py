import argparse
from dataclasses import dataclass
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from prophet import Prophet
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor


INPUT_LENGTH = 18
VALID_MONTHS = 48


@dataclass
class ModelSpec:
    name: str
    builder: callable


def load_monthly_series(csv_path: str) -> pd.Series:
    df = pd.read_csv(csv_path, sep=";", parse_dates=["date"], index_col="date")
    return df.interpolate(method="time").resample("ME").mean()["target"]


def month_one_hot(date: pd.Timestamp) -> list[float]:
    values = [0.0] * 12
    values[date.month - 1] = 1.0
    return values


def make_features(history: np.ndarray, next_date: pd.Timestamp, input_length: int = INPUT_LENGTH) -> np.ndarray:
    values = np.asarray(history, dtype=float)
    features = values[-input_length:].tolist()
    features.extend(month_one_hot(next_date))
    features.extend(
        [
            values[-3:].mean(),
            values[-6:].mean(),
            values[-12:].mean(),
            values[-12:].std(),
        ]
    )
    return np.asarray(features, dtype=float)


def build_training_data(series: pd.Series, input_length: int = INPUT_LENGTH) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for i in range(input_length, len(series)):
        history = series.iloc[i - input_length : i].values
        xs.append(make_features(history, series.index[i], input_length))
        ys.append(series.iloc[i])
    return np.asarray(xs), np.asarray(ys)


def recursive_forecast(
    model,
    history: pd.Series,
    future_index: pd.Index,
    input_length: int = INPUT_LENGTH,
) -> np.ndarray:
    values = list(history.values)
    predictions = []
    for date in future_index:
        features = make_features(np.asarray(values), date, input_length).reshape(1, -1)
        prediction = float(model.predict(features)[0])
        predictions.append(prediction)
        values.append(prediction)
    return np.asarray(predictions)


def seasonal_naive(history_and_valid: pd.Series, valid_index: pd.Index) -> np.ndarray:
    return history_and_valid.shift(12).reindex(valid_index).values


def score_predictions(true_values: np.ndarray, pred_values: np.ndarray) -> dict[str, float]:
    metrics = {
        "mse": mean_squared_error(true_values, pred_values),
        "rmse": mean_squared_error(true_values, pred_values) ** 0.5,
    }
    for horizon in (3, 6, 12):
        metrics[f"rmse_first_{horizon}m"] = mean_squared_error(
            true_values[:horizon], pred_values[:horizon]
        ) ** 0.5
    return metrics


def score_row(model_name: str, true_values: np.ndarray, pred_values: np.ndarray) -> dict[str, float]:
    metrics = score_predictions(true_values, pred_values)
    return {
        "model": model_name,
        "rmse_48m": metrics["rmse"],
        "rmse_12m": metrics["rmse_first_12m"],
        "rmse_6m": metrics["rmse_first_6m"],
        "rmse_3m": metrics["rmse_first_3m"],
    }


def evaluate_models(train_series: pd.Series, valid_series: pd.Series) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    x_train, y_train = build_training_data(train_series)
    full_series = pd.concat([train_series, valid_series])
    results = []
    forecast_store = {}

    naive_pred = seasonal_naive(full_series, valid_series.index)
    results.append(
        score_row(
            "seasonal_naive",
            valid_series.values,
            naive_pred,
        )
    )
    forecast_store["seasonal_naive"] = naive_pred

    ml_models = [
        ModelSpec("linear_ar_18", lambda: LinearRegression()),
        ModelSpec("ridge_ar_18", lambda: Ridge(alpha=0.1)),
        ModelSpec(
            "extra_trees_ar_18",
            lambda: ExtraTreesRegressor(
                n_estimators=600,
                random_state=42,
                min_samples_leaf=2,
            ),
        ),
        ModelSpec(
            "random_forest_ar_18",
            lambda: RandomForestRegressor(
                n_estimators=600,
                random_state=42,
                min_samples_leaf=2,
            ),
        ),
        ModelSpec(
            "xgboost_ar_18",
            lambda: XGBRegressor(
                n_estimators=400,
                learning_rate=0.03,
                max_depth=3,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                random_state=42,
            ),
        ),
        ModelSpec(
            "lightgbm_ar_18",
            lambda: LGBMRegressor(
                n_estimators=400,
                learning_rate=0.03,
                num_leaves=15,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                verbose=-1,
            ),
        ),
    ]

    for spec in ml_models:
        model = spec.builder()
        model.fit(x_train, y_train)
        predictions = recursive_forecast(model, train_series, valid_series.index)
        results.append(score_row(spec.name, valid_series.values, predictions))
        forecast_store[spec.name] = predictions

    autoreg = AutoReg(
        train_series,
        lags=[1, 2, 3, 6, 12],
        seasonal=True,
        old_names=False,
        trend="c",
    ).fit()
    autoreg_pred = autoreg.predict(
        start=len(train_series),
        end=len(train_series) + len(valid_series) - 1,
        dynamic=True,
    ).values
    results.append(
        score_row(
            "autoreg_lags=[1,2,3,6,12]",
            valid_series.values,
            autoreg_pred,
        )
    )
    forecast_store["autoreg_lags=[1,2,3,6,12]"] = autoreg_pred

    hw = ExponentialSmoothing(
        train_series,
        trend="add",
        seasonal="add",
        seasonal_periods=12,
    ).fit(optimized=True, use_brute=True)
    hw_pred = hw.forecast(len(valid_series)).values
    results.append(score_row("holt_winters_add_add", valid_series.values, hw_pred))
    forecast_store["holt_winters_add_add"] = hw_pred

    prophet_train = train_series.reset_index().rename(columns={"date": "ds", "target": "y"})
    prophet_model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.01,
        seasonality_prior_scale=5.0,
    )
    prophet_model.fit(prophet_train)
    future = prophet_model.make_future_dataframe(periods=len(valid_series), freq="ME")
    prophet_pred = prophet_model.predict(future).iloc[-len(valid_series) :]["yhat"].values
    results.append(score_row("prophet_cps=0.01", valid_series.values, prophet_pred))
    forecast_store["prophet_cps=0.01"] = prophet_pred

    sarimax = SARIMAX(
        train_series,
        order=(2, 0, 0),
        seasonal_order=(1, 1, 0, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    sarimax_pred = sarimax.forecast(len(valid_series)).values
    results.append(
        score_row(
            "sarimax_(2,0,0)(1,1,0,12)",
            valid_series.values,
            sarimax_pred,
        )
    )
    forecast_store["sarimax_(2,0,0)(1,1,0,12)"] = sarimax_pred

    ranked = pd.DataFrame(results).sort_values(["rmse_48m", "rmse_12m", "rmse_6m", "rmse_3m"])
    return ranked, forecast_store


def print_results(results: pd.DataFrame) -> None:
    print(results.to_string(index=False))


def plot_benchmark(results: pd.DataFrame, output_path: str) -> None:
    ranked = results.sort_values("rmse_48m").reset_index(drop=True)
    plt.figure(figsize=(12, 6))
    plt.barh(ranked["model"], ranked["rmse_48m"], color="steelblue")
    plt.gca().invert_yaxis()
    plt.xlabel("RMSE over the 48-month validation block")
    plt.ylabel("Model")
    plt.title("Model Ranking on Validation")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_best_validation_forecast(
    train_series: pd.Series,
    valid_series: pd.Series,
    best_model_name: str,
    best_predictions: np.ndarray,
    output_path: str,
) -> None:
    plt.figure(figsize=(13, 6))
    history = train_series.iloc[-60:]
    plt.plot(history.index, history.values, label="Train history", color="tab:blue", marker=".")
    plt.plot(valid_series.index, valid_series.values, label="Validation truth", color="tab:green", marker="o")
    plt.plot(
        valid_series.index,
        best_predictions,
        label=f"Validation forecast ({best_model_name})",
        color="tab:red",
        linestyle="--",
        marker="x",
    )
    plt.xlabel("Date")
    plt.ylabel("Target")
    plt.title("Best Model Against the Validation Block")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_metric_heatmap(results: pd.DataFrame, output_path: str) -> None:
    ranked = results.sort_values("rmse_48m").reset_index(drop=True)
    metric_labels = ["3 months", "6 months", "12 months", "48 months"]
    metric_cols = ["rmse_3m", "rmse_6m", "rmse_12m", "rmse_48m"]
    values = ranked[metric_cols].to_numpy()

    fig_height = max(5.5, 0.45 * len(ranked) + 2.0)
    plt.figure(figsize=(10, fig_height))
    image = plt.imshow(values, cmap="YlGnBu_r", aspect="auto")
    plt.colorbar(image, label="RMSE")
    plt.xticks(range(len(metric_labels)), metric_labels)
    plt.yticks(range(len(ranked)), ranked["model"])
    plt.title("Model Comparison Across Forecast Horizons")

    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            plt.text(col, row, f"{values[row, col]:.2f}", ha="center", va="center", color="black")

    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_top_model_grid(
    train_series: pd.Series,
    valid_series: pd.Series,
    results: pd.DataFrame,
    forecast_store: dict[str, np.ndarray],
    output_path: str,
    top_n: int = 6,
) -> None:
    ranked = results.sort_values("rmse_48m").head(top_n).reset_index(drop=True)
    cols = 2
    rows = ceil(len(ranked) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4.2 * rows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()

    history = train_series.iloc[-36:]
    y_min = min(history.min(), valid_series.min(), min(np.min(forecast_store[name]) for name in ranked["model"]))
    y_max = max(history.max(), valid_series.max(), max(np.max(forecast_store[name]) for name in ranked["model"]))
    padding = 0.08 * (y_max - y_min)

    for ax, (_, row) in zip(axes, ranked.iterrows()):
        model_name = row["model"]
        ax.plot(history.index, history.values, color="tab:blue", marker=".", linewidth=1.2, label="Train")
        ax.plot(valid_series.index, valid_series.values, color="tab:green", marker="o", linewidth=1.4, label="Truth")
        ax.plot(
            valid_series.index,
            forecast_store[model_name],
            color="tab:red",
            linestyle="--",
            marker="x",
            linewidth=1.4,
            label="Forecast",
        )
        ax.set_title(f"{model_name}\n48m RMSE = {row['rmse_48m']:.2f}")
        ax.grid(alpha=0.3)
        ax.set_ylim(y_min - padding, y_max + padding)

    for ax in axes[len(ranked) :]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Top Models on the Validation Block", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fit_best_model(series: pd.Series):
    x_train, y_train = build_training_data(series)
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model


def load_test_dates(csv_path: str) -> pd.DatetimeIndex:
    df = pd.read_csv(csv_path, sep=";", parse_dates=["date"])
    return pd.DatetimeIndex(df["date"])


def save_forecast(test_dates: pd.DatetimeIndex, predictions: np.ndarray, output_path: str) -> None:
    forecast_df = pd.DataFrame({"date": test_dates, "target": predictions})
    forecast_df.to_csv(output_path, sep=";", index=False, float_format="%.4f")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default="test.csv")
    parser.add_argument("--output", default="linear_forecast_test.csv")
    parser.add_argument("--benchmark-output", default="benchmark_results.csv")
    parser.add_argument("--benchmark-plot", default="benchmark_plot.png")
    parser.add_argument("--validation-plot", default="best_validation_forecast.png")
    parser.add_argument("--metric-heatmap", default="benchmark_heatmap.png")
    parser.add_argument("--top-model-grid", default="top_models_validation_grid.png")
    args = parser.parse_args()

    monthly_series = load_monthly_series(args.train)
    train_series = monthly_series.iloc[:-VALID_MONTHS]
    valid_series = monthly_series.iloc[-VALID_MONTHS:]

    print("Validation benchmark on the last 48 months")
    results, forecast_store = evaluate_models(train_series, valid_series)
    print_results(results)
    results.to_csv(args.benchmark_output, index=False)

    best_model_name = results.iloc[0]["model"]
    plot_benchmark(results, args.benchmark_plot)
    plot_best_validation_forecast(
        train_series,
        valid_series,
        best_model_name,
        forecast_store[best_model_name],
        args.validation_plot,
    )
    plot_metric_heatmap(results, args.metric_heatmap)
    plot_top_model_grid(train_series, valid_series, results, forecast_store, args.top_model_grid)

    best_model = fit_best_model(monthly_series)
    test_dates = load_test_dates(args.test)
    predictions = recursive_forecast(best_model, monthly_series, test_dates)
    save_forecast(test_dates, predictions, args.output)

    print()
    print("Saved forecast file:")
    print(f"  {args.output}")
    print("Saved benchmark file:")
    print(f"  {args.benchmark_output}")
    print("Saved benchmark plot:")
    print(f"  {args.benchmark_plot}")
    print("Saved validation plot:")
    print(f"  {args.validation_plot}")
    print("Saved metric heatmap:")
    print(f"  {args.metric_heatmap}")
    print("Saved top-model grid:")
    print(f"  {args.top_model_grid}")


if __name__ == "__main__":
    main()
