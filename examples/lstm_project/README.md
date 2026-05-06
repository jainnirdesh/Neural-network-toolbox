LSTM Example Project
====================

This example demonstrates a clean, modular TensorFlow/Keras LSTM pipeline suitable for time-series prediction or sequence tasks.

Features:
- Synthetic data generator (sine waves) if you don't have a dataset yet
- `preprocess_data()` and `create_sequences()` for LSTM-ready inputs
- `build_lstm_model()` using `Sequential` with LSTM, Dropout and Dense layers
- `train_model()` with EarlyStopping and ModelCheckpoint
- `evaluate_model()` with plots for loss and predicted vs actual

Run the example:

```bash
python examples/lstm_project/lstm_pipeline.py --help
```

Or run with defaults (synthetic sine data):

```bash
python examples/lstm_project/lstm_pipeline.py
```

If you have a CSV dataset, provide `--csv path/to/file.csv --col column_name`.
