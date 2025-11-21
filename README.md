# wutis_quantum_on_quantum

**WUTIS Semester Project WS 25/26**
Markowitz Portfolio Optimization using Quantum Neuronal Networks and Quantum Annealer.

---
## How to Use

### Install packages 
To start with the project, run `install.sh` script. Python version used: 3.11.14

### Fetch and Preprocess Data
Run the following scripts in sequence to fetch and preprocess the data:

1. **`fetch.py`**  
    Fetch historical market data from external APIs.
2. **`preprocess.py`**  
    Add indicators defined in `indicators.py` and merge data into `raw/merged_data.csv `.
3. **`datasets.py`**  
    Create datasets for covariance and returns prediction and save in .npz under `processed/qnn_datasets.npz`.

### QNN Training and Tuning 

1. **`encode.py`**  
    Encode the datasets into unitary QNN (data) layer. 
2. **`model.py`**  
    QNN model and quantum circuit definition
3. **`train.py`** 
    Train the model with predefined parameters
4. **`tuning.py`**  
    Tune model's hyperparameters with Optuna

### Alpaca API Configuration
Alpaca API is needed to fetch stock data from the market. 
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
├── qnn/                  # QNN training, tuning
├── source/               # Project sources: presentation, documentations etc.
└── README.md             # Project documentation
```