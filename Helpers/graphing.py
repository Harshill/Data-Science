import numpy as np
import matplotlib.pyplot as graph


def plot_multiple_scatters(x_col_names, y_col_names, df, show_graph=False, figsize=(8, 8)):
    if isinstance(y_col_names, str):
        y_col_names = len(x_col_names) * [y_col_names]
    else:
        assert len(x_col_names) == len(y_col_names), 'either same number of x and y columns or just one y column'

    fig = graph.figure(figsize=figsize)
    fig.subplots_adjust(hspace=1, wspace=0.2)

    for idx, (x_col, y_col) in enumerate(zip(x_col_names, y_col_names)):
        n_cols = 2
        n_rows = np.ceil(len(x_col_names) / 2)

        graph.subplot(n_rows, n_cols, idx + 1)
        graph.scatter(df[x_col], df[y_col])
        graph.xlabel(x_col, labelpad=15)
        graph.ylabel(y_col)

    if show_graph:
        graph.show()