import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
from lifelines import WeibullAFTFitter
from lifelines import LogNormalAFTFitter

def censored_regression(data):
    categories = ["meme","ort","person","politik","text"]

    df = data.copy()

    # 1. Define censoring
    df["event_observed"] = ~df["duration"].isin([5.0, 15.0])  

    # 2. Split categoryCode into individual binary columns
    # Ensure it's a 5-character string
    df["categoryCode"] = df["categoryCode"].astype(str).str.zfill(5)
    
    for i in range(5):  # five digits
        df[categories[i]] = df["categoryCode"].str[i].astype(int)

    # 3. Fit Weibull AFT model
    aft = WeibullAFTFitter()
    aft.fit(
        df,
        duration_col="duration",
        event_col="event_observed",
        formula="meme + ort + person + politik + text + words"
    )

    # 4. Show results
    aft.print_summary()
    aft.plot()

censored_regression(pd.read_csv("EWS/code/analysis/duration/view_durations.csv"))

