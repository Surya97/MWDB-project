import plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


class VisualizeClusters:
    def __init__(self, dataset_features, cluster_details):
        self.dataframe = pd.DataFrame(list(dataset_features.values()))
        self.cluster_details = cluster_details
        self.dataframe["Cluster"] = list(self.cluster_details.values())
        # print("Dataframe head after adding cluster details")
        # print(self.dataframe.head())
        self.pca = PCA(n_components=2)
        self.PCA_dataframe = pd.DataFrame(self.pca.fit_transform(self.dataframe.drop(["Cluster"], axis=1)))
        self.PCA_dataframe.columns = ["PC1", "PC2"]
        self.dataframe = pd.concat([self.dataframe, self.PCA_dataframe], axis=1, join='inner')
        # print("Dataframe after adding PCs")
        # print(self.dataframe.head())
        self.cluster_data = list()
        self.trace = []

        self.cluster_image_map = {}

        for image_id, cluster in cluster_details.items():
            if cluster in self.cluster_image_map:
                self.cluster_image_map[cluster].append(image_id)
            else:
                self.cluster_image_map[cluster] = [image_id]
        self.colors = ['rgb(127,0,0)', 'rgb(140,105,105)', 'rgb(217,76,54)', 'rgb(51,20,0)', 'rgb(127,70,32)',
                       'rgb(64,54,48)', 'rgb(217,116,0)', 'rgb(255,170,0)', 'rgb(102,68,0)', 'rgb(217,181,108)',
                       'rgb(195,230,57)', 'rgb(133,140,105)', 'rgb(93,140,0)', 'rgb(34,64,0)', 'rgb(57,230,57)',
                       'rgb(172,230,172)', 'rgb(0,128,85)', 'rgb(64,255,191)', 'rgb(121,242,234)', 'rgb(38,74,77)',
                       'rgb(0,194,242)', 'rgb(35,119,140)', 'rgb(26,66,102)', 'rgb(172,203,230)', 'rgb(0,92,230)',
                       'rgb(105,119,140)', 'rgb(115,145,230)', 'rgb(0,19,140)', 'rgb(0,0,255)', 'rgb(0,0,217)',
                       'rgb(64,64,128)', 'rgb(28,13,51)', 'rgb(163,54,217)', 'rgb(210,172,230)', 'rgb(119,0,128)',
                       'rgb(255,0,204)', 'rgb(89,67,85)', 'rgb(230,115,191)', 'rgb(166,0,88)', 'rgb(102,0,41)',
                       'rgb(255,64,115)', 'rgb(166,83,94)', 'rgb(242,182,190)']
        self.set_cluster_data()

    def set_cluster_data(self):
        # print("Cluster data length", len(self.cluster_data))
        for i in range(len(set(self.cluster_details.values()))):
            # print("Cluster ", str(i))
            # print(self.dataframe[self.dataframe["Cluster"] == i])
            self.cluster_data.append(self.dataframe[self.dataframe["Cluster"] == i])

        # print("Cluster data shape", len(self.cluster_data), len(self.cluster_data[0]))

    def plot(self):
        self.trace = [0]*len(self.cluster_data)
        for i in range(len(self.trace)):
            self.trace[i] = go.Scatter(x=self.cluster_data[i]["PC1"],
                                       y=self.cluster_data[i]["PC2"],
                                       mode="markers",
                                       name="Cluster "+str(i),
                                       marker=dict(color=self.colors[i%len(self.colors)]),
                                       text=self.cluster_image_map[i])

        title = "Cluster visualisation in 2D using PCA"
        layout = dict(title=title,
                      xaxis=dict(title="PC1", ticklen=5, zeroline=False),
                      yaxis=dict(title='PC2', ticklen=5, zeroline=False))

        fig = dict(data=self.trace, layout=layout)

        iplot(fig)


