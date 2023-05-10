class DoubleExperiment:
    def __init__(self, first, second, scene):
        self.first = first
        self.second = second
        self.scene = scene


class SingleExperiment:
    def __init__(self, vals, scene):
        self.vals = vals
        self.scene = scene


def text_to_df(text):
    import pandas as pd
    import numpy as np

    lines = text.split('\n')
    items = [line.split('\t') for line in lines]
    # add column names to df
    columns = ["Model", "2.5% (pretrained)", "5% (pretrained)", "25% (pretrained)", "50% (pretrained)",
               "2.5% (random)", "5% (random)", "25% (random)", "50% (random)"]
    df = pd.DataFrame(items, columns=columns)
    # Convert empty cells to NaN
    df = df.replace(r'^\s*$', np.nan, regex=True)
    return df

def experiment_to_csv(exp):
    dfs = [text_to_df(x.first) for x in exp.experiments] + [text_to_df(x.second) for x in exp.experiments]
    return [df.to_csv() for df in dfs]
