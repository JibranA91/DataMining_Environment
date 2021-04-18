

def correl_matrix(dataframe):

    import seaborn as sb
    from io import BytesIO
    import base64
    import matplotlib.pyplot as plt

    img = BytesIO()
    sb.set_style("dark")

    dataframe_factors = dataframe.apply(
        lambda x: x.factorize()[0])
    corr_matrix = dataframe_factors.corr()
    sb.heatmap(corr_matrix)

    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    return plot_url
