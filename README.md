# wutis_quantum_on_quantum

**WUTIS Semester Project WS 25/26**
Markowitz Portfolio Optimization using Quantum Neuronal Networks and Quantum Annealer.

---s
## How to Use

### Fetch and Preprocess Data
Run the following scripts in sequence to fetch and preprocess the data:

1. **`01_fetch.py`**  
    Fetch historical market data from external APIs.
2. **`02_preprocess.py`**  
    Add indicators defined in `indicators.py` and merge data into `raw/merged_data.csv `.
3. **`03_datasets.py`**  
    Create datasets for covariance and returns prediction and save in .npz under `processed/qnn_datasets.npz`.

### Alpaca API Configuration
To use the Alpaca API for historical data, define a `config.yaml` file in the `config/` directory as follows:

```yaml
alpaca_api: 
  secret_key: 'your_secret_key'
  api_key: 'your_api_key'
  base_url: 'https://paper-api.alpaca.markets' # Example URL
```

### Project Structure: 
```
wutis-quantum/
├── analysis/             # Data visualizations, plots,...
├── config/               # Configuration files
├── data/                 # Data methods
├── source/               # Project sources: presentation, documentations etc.
└── README.md             # Project documentation
```