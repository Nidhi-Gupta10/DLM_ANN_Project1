import pandas as pd

def model_summary_to_df(model):
    layers = []
    for layer in model.layers:
        try:
            output_shape = layer.output_shape
        except AttributeError:
            output_shape = "N/A"

        try:
            params = layer.count_params()
        except Exception:
            params = "N/A"

        layers.append({
            "Name": layer.name,
            "Class": layer.__class__.__name__,
            "Output Shape": output_shape,
            "Params": params
        })
    return pd.DataFrame(layers)
