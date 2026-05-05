import pandas as pd
from fastapi import HTTPException

def make_predictions(model, data):

    # If single object, convert to list
    if isinstance(data, dict):
        data = [data]

    df = pd.DataFrame(data)

    try:
        predictions = model.predict(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return predictions.tolist()