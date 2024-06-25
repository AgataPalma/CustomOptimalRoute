import osmnx as ox
import pandas as pd
import networkx as nx
from PyQt5.QtCore import QVariant
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QMessageBox
from qgis.core import QgsProject, QgsVectorLayer, QgsFeature, QgsField, QgsGeometry, QgsPointXY, \
    QgsCategorizedSymbolRenderer, QgsRendererCategory, QgsMarkerSymbol, QgsMessageLog, Qgis, QgsFields
import osmnx.truncate as ox_truncate
from sklearn.neighbors import sort_graph_by_row_values


class ClusterAnalysis:
    def __init__(self, iface):
        self.iface = iface

    def run_dbscan_clustering(self, pois, place, eps=300, minpts=4):
        # The graph_from_place creates a road network graph for the specified place using OSMnx.
        # The CRS of this graph is typically in UTM (meters).
        G = ox.graph_from_place(place, network_type='drive')

        # This ensures that only the largest connected component of the graph is used.
        G = ox_truncate.largest_component(G, strongly=True)

        # Filter the POIs to only include points and then project the POIs to the CRS of the graph.
        # The CRS of the graph (G.graph['crs']) is obtained from the graph’s metadata and used for projecting the POIs
        pois = pois[pois.geometry.type == 'Point']
        pois = pois.to_crs(G.graph['crs'])

        # Find the nearest nodes in the graph for each POI
        pois['nn'] = ox.nearest_nodes(G, X=pois.geometry.x, Y=pois.geometry.y)

        # Get unique nodes from the nearest nodes
        nodes_unique = pd.Series(pois['nn'].unique())
        nodes_unique.index = nodes_unique.values

        # Extract x and y coordinates for each unique node
        x = nodes_unique.map(lambda x: G.nodes[x]['x'])
        y = nodes_unique.map(lambda x: G.nodes[x]['y'])
        df = pd.DataFrame({'x': x, 'y': y}, index=nodes_unique)

        # Compute the pairwise Euclidean distance matrix
        dist_matrix = squareform(pdist(X=df, metric='euclidean'))
        df_dist_matrix = pd.DataFrame(data=dist_matrix, columns=df.index.values, index=df.index.values)
        node_euclidean_dists = df_dist_matrix.stack()

        # Define a function to compute the network distance matrix
        def network_distance_matrix(u):
            nearby_nodes = node_euclidean_dists[u][node_euclidean_dists[u] < 0.005].index
            net_dists = [nx.dijkstra_path_length(G, source=u, target=v, weight='length') for v in nearby_nodes]
            return pd.Series(data=net_dists, index=nearby_nodes).reindex(nodes_unique).fillna(eps + 1).astype(int)

        # Apply the network distance matrix function to each unique node
        node_dm = nodes_unique.apply(network_distance_matrix)
        node_dm[node_dm == 0] = 1
        node_dm[node_dm > eps] = 0
        node_dm = csr_matrix(node_dm)

        # Sort the distance matrix
        node_dm_sorted = sort_graph_by_row_values(node_dm, warn_when_not_sorted=False)
        node_dm_sorted.setdiag(0)

        # Perform DBSCAN clustering
        db = DBSCAN(eps=eps, min_samples=minpts, metric='precomputed')
        cluster_labels = db.fit_predict(node_dm_sorted)

        # Validate the clustering results
        self.validate_clustering(node_dm_sorted, cluster_labels)

        # Map nodes to cluster labels
        node_clusters = {node: label for node, label in zip(nodes_unique.index, cluster_labels)}
        pois['cluster'] = pois['nn'].map(lambda x: node_clusters[x])

        # Exclude noise points
        clustered_pois = pois[pois['cluster'] != -1]

        if not clustered_pois.empty:
            # largest cluster
            largest_cluster_label = clustered_pois['cluster'].value_counts().idxmax()
            largest_cluster = clustered_pois[clustered_pois['cluster'] == largest_cluster_label]
            largest_cluster_centroid = largest_cluster.geometry.unary_union.centroid

            # Sample up to 10 points from the largest cluster
            if len(largest_cluster) > 0:
                largest_cluster_sample = largest_cluster.sample(min(len(largest_cluster), 10))
            else:
                QgsMessageLog.logMessage("No points in the largest cluster", "OptimalCustomRoute", Qgis.Critical)
                return None, None, None, None

            # densest cluster
            def compute_cluster_density(cluster):
                area = cluster.geometry.unary_union.convex_hull.area
                if area > 0:
                    density = len(cluster) / area
                    return density
                else:
                    return 0

            cluster_densities = clustered_pois.groupby('cluster').apply(compute_cluster_density)
            most_dense_cluster_label = cluster_densities.idxmax()
            most_dense_cluster = clustered_pois[clustered_pois['cluster'] == most_dense_cluster_label]
            densest_cluster_centroid = most_dense_cluster.geometry.unary_union.centroid

        else:
            QMessageBox.information(self.iface.mainWindow(), "Cluster Info", "No clusters found")
            return None, None

        # Save clustered POIs to a new QGIS layer
        self.save_clustered_pois_to_layer(clustered_pois[['geometry', 'name', 'cluster']], "clustered_pois")
        # Visualize the clusters in QGIS
        self.visualize_clusters(clustered_pois)
        return largest_cluster_centroid, densest_cluster_centroid, largest_cluster_sample

    @staticmethod
    def validate_clustering(node_dm_sorted, cluster_labels):
        # Silhouette score
        if len(set(cluster_labels)) > 1:
            silhouette_avg = silhouette_score(node_dm_sorted, cluster_labels, metric='precomputed')
            QgsMessageLog.logMessage(f'Silhouette Score: {silhouette_avg}', 'OptimalCustomRoute', Qgis.Info)

            # Davies-Bouldin Index
            try:
                dbi_score = davies_bouldin_score(node_dm_sorted.toarray(), cluster_labels)
                QgsMessageLog.logMessage(f'Davies-Bouldin Index: {dbi_score}', 'OptimalCustomRoute', Qgis.Info)
            except ValueError as e:
                QgsMessageLog.logMessage(f'Davies-Bouldin Index could not be computed: {e}', 'OptimalCustomRoute',
                                         Qgis.Warning)
        else:
            QgsMessageLog.logMessage('Silhouette Score: Cannot be computed with only one cluster', 'OptimalCustomRoute',
                                     Qgis.Info)
            QgsMessageLog.logMessage('Davies-Bouldin Index: Cannot be computed with only one cluster',
                                     'OptimalCustomRoute', Qgis.Info)

    @staticmethod
    def visualize_clusters(clustered_pois):
        cluster_layer = QgsVectorLayer('Point?crs=EPSG:4326', 'Clustered POIs', 'memory')
        provider = cluster_layer.dataProvider()

        provider.addAttributes([
            QgsField('name', QVariant.String),
            QgsField('cluster', QVariant.Int)
        ])
        cluster_layer.updateFields()

        # Create features for each POI in the clustered POIs
        features = []
        for idx, row in clustered_pois.iterrows():
            # Create a new feature and set its geometry and attributes
            feature = QgsFeature()
            feature.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(row.geometry.x, row.geometry.y)))
            feature.setAttributes([row['name'], row['cluster']])
            features.append(feature)

        # Add features to the layer
        provider.addFeatures(features)
        cluster_layer.updateExtents()
        QgsProject.instance().addMapLayer(cluster_layer)

        # Define the categories for the clusters
        unique_clusters = clustered_pois['cluster'].unique()
        categories = []

        for cluster in unique_clusters:
            symbol = QgsMarkerSymbol.createSimple({'name': 'circle', 'size': '2'})
            if cluster == -1:
                symbol.setColor(QColor(128, 128, 128))  # Grey color for noise --- there should not be any grey point
            else:
                symbol.setColor(QColor.fromHsv((cluster * 137) % 360, 255, 255))  # Unique color for each cluster
            category = QgsRendererCategory(str(cluster), symbol, f'Cluster {cluster}')
            categories.append(category)

        # Create a categorized symbol renderer
        renderer = QgsCategorizedSymbolRenderer('cluster', categories)
        cluster_layer.setRenderer(renderer)
        cluster_layer.triggerRepaint()

    def save_clustered_pois_to_layer(self, clustered_pois, layer_name):
        # Define fields for the new layer
        fields = QgsFields()
        fields.append(QgsField('id', QVariant.Int))
        fields.append(QgsField('cluster', QVariant.Int))

        # Create a new memory layer for clustered POIs
        layer = QgsVectorLayer('Point?crs=EPSG:4326', layer_name, 'memory')
        layer_data_provider = layer.dataProvider()

        # Add fields to the layer
        layer_data_provider.addAttributes(fields)
        layer.updateFields()

        # Create features for each POI in the clustered POIs
        features = []
        for idx, row in clustered_pois.iterrows():
            feature = QgsFeature()
            feature.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(row.geometry.x, row.geometry.y)))
            feature.setAttributes([idx, row['cluster']])
            features.append(feature)

        # Add features to the layer
        layer_data_provider.addFeatures(features)

        # Add the new layer to the QGIS project and refresh the map canvas
        QgsProject.instance().addMapLayer(layer)
        self.iface.mapCanvas().refresh()
