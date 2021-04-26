# Plotting Graphs

def sortedgroupedbar(ax, x, y, groupby, data=None, is_data_grouped=0, xlabelrotation=45, width=0.8, **kwargs):
    from io import BytesIO
    import base64
    import matplotlib.pyplot as plt
    import numpy as np
    img = BytesIO()

    if groupby == "Don't Split":
        data = data[[x]]
        data_count = data.count()[0]
        data = data.groupby([x])[x].agg('count').to_frame('count').reset_index()
        data['percentage'] = round((data['count'] / data_count) * 100.0, 2)
        data = data.sort_values(by=['count'], ascending=False).reset_index()
        del [data['index']]
        ax.bar(data[x], data[y])
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.tick_params(axis='x', labelrotation=xlabelrotation)
    else:

        if is_data_grouped == 0:
            data = data[[x, groupby]]
            data_count = data.count()[1]
            data = data.groupby([x, groupby])[x].agg('count').to_frame('count').reset_index()
            data['percentage'] = round((data['count'] / data_count) * 100.0, 2)

        order = np.zeros(len(data))
        df = data.copy()
        for xi in np.unique(df[x].values):
            group = data[df[x] == xi]
            a = group[y].values
            b = sorted(np.arange(len(a)), key=lambda x: a[x], reverse=True)
            c = sorted(np.arange(len(a)), key=lambda x: b[x])
            order[data[x] == xi] = c
        df["order"] = order
        u, df["ind"] = np.unique(df[x].values, return_inverse=True)
        step = width / len(np.unique(df[groupby].values))
        for xi, grp in df.groupby(groupby):
            ax.bar(grp["ind"] - width / 2. + grp["order"] * step + step / 2.,
                   grp[y], width=step, label=xi, **kwargs)
        ax.legend(title=groupby)
        ax.set_xticks(np.arange(len(u)))
        ax.set_xticklabels(u)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.tick_params(axis='x', labelrotation=xlabelrotation)

    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    return plot_url
