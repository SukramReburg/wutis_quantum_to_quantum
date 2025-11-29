import numpy as np
import pandas as pd

class Indicator(): 
    def __init__(self, data, name=None): 
        self.data = data 
        self.name = name

    def calculate(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def add_indicator(self):
        result = self.calculate()
        if isinstance(result, pd.DataFrame):
            for col in result.columns:
                self.data[col] = result[col]
        else:
            self.data[self.name] = result
        return self.data

class RSI(Indicator):
    def __init__(self, data, period=14, name='rsi'):
        super().__init__(data, name)
        self.period = period

    def calculate(self):
        delta = self.data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
class Volatility(Indicator):
    def __init__(self, data, period, name='vola'):
        super().__init__(data, name)
        self.period = period

    def calculate(self):
        log_returns = np.log(self.data['close'] / self.data['close'].shift(1))
        volatility = log_returns.rolling(window=self.period).std() # * np.sqrt(252)  # Annualized volatility
        return volatility
    
class LogReturns(Indicator):
    def __init__(self, data, name='log'):
        super().__init__(data, name)

    def calculate(self):
        return np.log(self.data['close'] / self.data['close'].shift(1))
    
class VolumeChange(Indicator):
    def __init__(self, data, name='volume_change'):
        super().__init__(data, name)

    def calculate(self):
        return self.data['volume'].pct_change()
        volume = pd.to_numeric(self.data['volume'], errors='coerce')

        # Avoid infinite jumps caused by zero volume days (e.g., halts or missing
        # prints) by treating zeros as missing before computing the change.
        volume.replace(0, np.nan, inplace=True)

        change = volume.pct_change()
        return change
    

indicators_impl = {'rsi': RSI, 
                  'vola': Volatility,
                  'log': LogReturns,
                  'volume_change': VolumeChange
                  }