# wutis_quantum_to_quantum

**WUTIS Semester Project WS 25/26**
Markowitz Portfolio Optimization using Quantum Neuronal Networks and Quantum Annealer.

---
## How to Use

### 1. Fetch and Preprocess Data
Run the following scripts in sequence to fetch and preprocess the data:

1. **`fetch.py`**  
    Fetch historical market data from external APIs.

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
├── config/               # Configuration files
├── data/                 # Data methods
├── source/               # Project sources: presentation, documentations etc.
└── README.md             # Project documentation
```