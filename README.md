# short-horizon-stock-forecasting

## Overview
Reproducible walk-forward evaluation of price-only stock forecasting models (GBM, ARIMA, naive baselines) at a 30-day horizon.

This project provides a comprehensive framework for comparing different time-series forecasting approaches on stock price data. The focus is on short-term (30-day) predictions using only historical price information, with rigorous walk-forward validation to simulate real-world trading scenarios.

## Models
The following forecasting models are implemented and evaluated:

- **Gradient Boosting Machine (GBM)**: Tree-based ensemble method for non-linear pattern recognition
- **ARIMA**: Autoregressive Integrated Moving Average for capturing linear temporal dependencies
- **Naive Baselines**: Simple forecasting methods (e.g., last-value, moving average) for benchmark comparison

## Methodology
The project employs a walk-forward validation strategy to ensure realistic performance evaluation:

1. **Data Preparation**: Historical stock price data is collected and preprocessed
2. **Feature Engineering**: Price-based features are extracted for model training
3. **Walk-Forward Validation**: Models are trained on expanding windows and evaluated on subsequent 30-day periods
4. **Performance Metrics**: Models are compared using standard forecasting metrics (RMSE, MAE, MAPE, etc.)
5. **Statistical Analysis**: Results are analyzed for significance and robustness

## Results Summary
Performance results and model comparisons will be reported in the `outputs/` directory:

- Forecasting accuracy metrics for each model
- Statistical comparison of model performance
- Visualization of predictions vs. actual values
- Analysis of model behavior across different market conditions

## Reproducibility
To reproduce the results:

1. Clone this repository
2. Install dependencies (Python environment requirements will be documented)
3. Run the main evaluation pipeline
4. Results will be generated in the `outputs/` directory

Detailed setup and execution instructions will be provided in the project documentation.

## Outputs
All experimental results are organized in the `outputs/` directory:

- `outputs/logs/`: Execution logs and model training details
- `outputs/plots/`: Visualizations and charts
- `outputs/tables/`: Numerical results and performance metrics

Note: Output files are excluded from version control due to size. Use the provided scripts to regenerate results.

## Disclaimer
This project is for educational and research purposes only. The models and results presented here should not be used for actual trading or investment decisions. Stock market predictions are inherently uncertain, and past performance does not guarantee future results. Always consult with qualified financial advisors before making investment decisions.
