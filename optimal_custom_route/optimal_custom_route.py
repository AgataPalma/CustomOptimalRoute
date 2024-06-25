import numpy as np
np.float = float

import os
import osmnx as ox
import openrouteservice
from PyQt5.QtGui import QIcon
import pandas as pd
import geopandas as gpd
from openrouteservice.exceptions import ApiError
from PyQt5 import uic
from PyQt5.QtWidgets import QMessageBox, QDialog, QCheckBox, QFileDialog, QAction
from PyQt5.QtCore import Qt, QVariant, QThread, QCoreApplication
from qgis.core import QgsRasterLayer, QgsRectangle, QgsPointXY, QgsVectorLayer, QgsField, \
    QgsFeature, QgsGeometry, QgsProject, QgsMessageLog, Qgis, QgsCategorizedSymbolRenderer, \
    QgsRendererCategory, QgsMarkerSymbol
from qgis.utils import iface
from shapely.geometry import Point
import pyproj
import gpxpy
import gpxpy.gpx
from .RouteAnimator import RouteAnimator
from .clustering import ClusterAnalysis
from .exportpdf import PdfExport
from geopy.distance import geodesic


# OpenRouteService API key
ORS_API_KEY = "5b3ce3597851110001cf62485cb3917e7acc4417a3ab2fdab472edbd"
ors_client = openrouteservice.Client(key=ORS_API_KEY)

# UI files
Ui_CityInputDialog, _ = uic.loadUiType(os.path.join(os.path.dirname(__file__), 'city_input_dialog.ui'))
Ui_DetailsInputDialog, _ = uic.loadUiType(os.path.join(os.path.dirname(__file__), 'details_input_dialog.ui'))

# Note: We'll use the "provider" in our plugin. In QGIS, a "provider" is an interface that allows access to
# the underlying data of a vector layer. The data provider is responsible for reading, writing, and managing the
# data stored in the layer. It acts as a bridge between the QGIS layer and the data source, providing
# the necessary functions to manipulate the data.

class CityInputDialog(QDialog, Ui_CityInputDialog):
    def __init__(self):
        super().__init__()
        self.setupUi(self)


class DetailsInputDialog(QDialog, Ui_DetailsInputDialog):
    def __init__(self):
        super().__init__()
        self.addressInputDisabled = None
        self.setupUi(self)
        self.toggle_save_button(self.checkboxStartingPoint.checkState())
        self.toggle_dbscan_fields(self.checkboxDBSCAN.checkState())
        self.checkboxStartingPoint.stateChanged.connect(self.toggle_save_button)
        self.comboBoxPOI.currentIndexChanged.connect(self.toggle_category_checkboxes)
        self.checkboxDBSCAN.stateChanged.connect(self.toggle_dbscan_fields)

        self.categories = {
            'Tourism': ['hotel', 'museum'],
            'Amenity': ['restaurant', 'bar', 'cafe'],
            'Shop': ['mall']
        }

        self.category_checkboxes = {}

        # Dynamically create checkboxes for each subcategory
        row = 0
        col = 0
        for category, subcategories in self.categories.items():
            for subcategory in subcategories:
                checkbox = QCheckBox(subcategory)
                self.layoutCategories.addWidget(checkbox, row, col)
                self.category_checkboxes[subcategory] = checkbox
                row += 1
                if row > 2:
                    row = 0
                    col += 1

        self.toggle_category_checkboxes(self.comboBoxPOI.currentIndex())

        # Connect buttons for animation
        self.startAnimationButton.clicked.connect(self.start_animation)
        self.stopAnimationButton.clicked.connect(self.stop_animation)
        self.route_animator = None

    def start_animation(self):
        if self.route_animator:
            self.route_animator.start_animation()

    def stop_animation(self):
        if self.route_animator:
            self.route_animator.stop_animation()

    def toggle_save_button(self, state):
        visible = state == Qt.Checked
        self.addressLabel.setVisible(visible)
        self.addressInput.setVisible(visible)
        self.saveStartingPointButton.setVisible(visible)

        if not visible and getattr(self, 'addressInputDisabled', False):
            self.addressInput.setDisabled(False)
            self.addressInputDisabled = False
            # use_cluster_centroid_as_start = True

    def toggle_category_checkboxes(self, index):
        if self.comboBoxPOI.currentText() == "All":
            self.set_categories_state(True, False)
        else:
            self.set_categories_state(False, True)

    def set_categories_state(self, check_state, enabled):
        for checkbox in self.category_checkboxes.values():
            checkbox.setChecked(check_state)
            checkbox.setEnabled(enabled)

    def get_selected_categories(self):
        selected_categories = [checkbox.text() for checkbox in self.category_checkboxes.values() if
                               checkbox.isChecked()]
        return selected_categories

    def toggle_dbscan_fields(self, state):
        visible = state == Qt.Checked
        self.epsLabel.setVisible(visible)
        self.epsInput.setVisible(visible)
        self.minPtsLabel.setVisible(visible)
        self.minPtsInput.setVisible(visible)
        if visible:
            self.epsInput.setText(str(self.get_dbscan_values()[0]))  # Default value
            self.minPtsInput.setText(str(self.get_dbscan_values()[1]))  # Default value

    def get_dbscan_values(self):
        eps = int(self.epsInput.text()) if self.epsInput.text() else 300
        min_pts = int(self.minPtsInput.text()) if self.minPtsInput.text() else 4
        return eps, min_pts


class OptimalCustomRoute(QThread):
    def __init__(self, iface):
        QThread.__init__(self)
        self.iface = iface
        self.cluster_analysis = ClusterAnalysis(iface)
        self.export_pdf = PdfExport(self.iface.mapCanvas())
        self.mainWindow = None
        self.plugin_dir = os.path.dirname(__file__)
        self.actions = []
        self.menu = self.tr(u'&OptimalCustomRoute')
        self.first_start = True
        self.canvas = self.iface.mapCanvas()
        self.point_layer_name = "Starting Point"
        self.saved_point = False
        self.route_layer = None
        self.route_animator = None
        self.route = None

    def tr(self, message):
        return QCoreApplication.translate('OptimalCustomRoute', message)

    def add_action(self, icon_path, text, callback, enabled_flag=True, add_to_menu=True, add_to_toolbar=True,
                   status_tip=None, whats_this=None, parent=None):
        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)

        if status_tip is not None:
            action.setStatusTip(status_tip)

        if whats_this is not None:
            action.setWhatsThis(whats_this)

        if add_to_toolbar:
            self.iface.addToolBarIcon(action)

        if add_to_menu:
            self.iface.addPluginToMenu(self.menu, action)

        self.actions.append(action)
        return action

    def initGui(self):
        if not self.first_start:
            return

        icon_path = os.path.join(os.path.dirname(__file__), "icon.png")
        self.add_action(
            icon_path,
            text=self.tr(u'OptimalCustomRoute'),
            callback=self.run,
            parent=self.iface.mainWindow()
        )

        self.first_start = False

    def unload(self):
        for action in self.actions:
            self.iface.removePluginMenu(self.tr(u'&OptimalCustomRoute'), action)
            self.iface.removeToolBarIcon(action)
        self.actions = []

    def run(self):
        self.cityDlg = CityInputDialog()
        self.cityDlg.nextButton.clicked.connect(self.handle_city_input)
        self.cityDlg.show()

    def handle_city_input(self):
        try:
            # Get the city name from the input dialog
            city = self.cityDlg.cityInput.text()
            # Check if the city name is empty
            if not city:
                QMessageBox.warning(self.cityDlg, "Input Error", "Please enter a city name.")
                return

            # Disable the next button to prevent multiple submissions
            self.cityDlg.nextButton.setEnabled(False)
            result = self.retrieve_city_data(city)

            # If city data is successfully retrieved, proceed with the next step
            if result:
                self.on_city_data_retrieved(result)
        except Exception as e:
            QMessageBox.warning(self.cityDlg, "Input Error", f"Please enter a valid city name. Error: {e}")
            # Re-enable the next button in case of an error
            self.cityDlg.nextButton.setEnabled(True)
            return None

    def retrieve_city_data(self, city):
        # Geocode the city to a GeoDataFrame
        gdf = ox.geocode_to_gdf(city)

        # Check if the GeoDataFrame is empty
        if gdf.empty:
            QMessageBox.warning(self.cityDlg, "Data Error", "Geocoding failed. No data returned for the city.")
            return None

        # Get the bounding box of the city
        bbox = gdf.unary_union.bounds

        # Project the GeoDataFrame to EPSG:3857 (Web Mercator)
        gdf_projected = gdf.to_crs(epsg=3857)

        # Calculate the centroid of the projected city geometry
        city_centroid_projected = gdf_projected.geometry.centroid.iloc[0]

        # Transform the centroid coordinates back to EPSG:4326 (WGS 84)
        transformer = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        city_point_x, city_point_y = transformer.transform(city_centroid_projected.x, city_centroid_projected.y)

        # Create a GeoSeries for the city centroid point
        city_point = gpd.GeoSeries([Point(city_point_x, city_point_y)], crs="EPSG:4326").iloc[0]

        # Import the city map
        self.import_city_map()

        # Return the GeoDataFrame, bounding box, and city centroid point
        return gdf, bbox, city_point

    @staticmethod
    def import_city_map():
        # Define the URL for the OSM Standard tile service
        url_with_params = 'type=xyz&url=https://tile.openstreetmap.org/{z}/{x}/{y}.png'

        # Create a raster layer using the URL
        layer = QgsRasterLayer(url_with_params, "OSM Standard", "wms")

        # Check if the layer is valid
        if not layer.isValid():
            raise Exception("Failed to load the OSM Standard layer")

        # Add the layer to the QGIS project
        QgsProject.instance().addMapLayer(layer)

    def on_city_data_retrieved(self, result):
        # Check if the result is valid
        if result:
            # Unpack the result into class attributes
            self.gdf, self.bbox, self.city_point = result

            # Close the city input dialog
            self.cityDlg.close()

            # Zoom to the city's bounding box
            self.zoom_to_city_bbox()

            # Show the details input dialog
            self.show_details_dialog()
        else:
            # Re-enable the next button in case of an invalid result
            self.cityDlg.nextButton.setEnabled(True)

    # Zoom in to the bbox
    def zoom_to_city_bbox(self):
        xmin, ymin, xmax, ymax = self.bbox
        rect = QgsRectangle(xmin, ymin, xmax, ymax)
        self.canvas.setExtent(rect)
        self.canvas.refresh()

    def show_details_dialog(self):
        self.detailsDlg = DetailsInputDialog()
        self.detailsDlg.selectOutputPathButton.clicked.connect(self.select_output_path)
        self.detailsDlg.generateRouteButton.clicked.connect(self.generate_route)
        self.detailsDlg.saveStartingPointButton.clicked.connect(self.save_or_replace_point)
        self.detailsDlg.show()

    def select_output_path(self):
        output_path = QFileDialog.getSaveFileName(self.detailsDlg, "Save Route as PDF", "", "*.pdf")[0]
        if output_path:
            self.detailsDlg.outputInput.setText(output_path)

    def generate_route(self):
        # Get the selected category from the combo box
        category = self.detailsDlg.comboBoxPOI.currentText()

        # Get the output path from the input field
        output_path = self.detailsDlg.outputInput.text()

        # Check if the output path is empty
        if not output_path:
            QMessageBox.warning(self.detailsDlg, "Input Error", "Please fill all fields.")
            return

        # Get the address from the input field and strip any leading/trailing whitespace
        address = self.detailsDlg.addressInput.text().strip()
        use_cluster_centroid_as_start = False
        use_bbox_center_as_start = False

        # Determine the starting point if no address is provided and no point is saved
        if not address and not self.saved_point:
            # address = (self.city_point.y, self.city_point.x)
            use_cluster_centroid_as_start = True
            use_bbox_center_as_start = False
            self.saved_point = False

        # Disable the generate route button to prevent multiple submissions
        self.detailsDlg.generateRouteButton.setEnabled(False)

        # Get DBSCAN parameters if the checkbox is checked
        if self.detailsDlg.checkboxDBSCAN.isChecked():
            eps, min_pts = self.detailsDlg.get_dbscan_values()
        else:
            eps, min_pts = 300, 4

        # Process route generation
        message = self.process_route_generation(
            category, output_path, address, eps, min_pts, use_cluster_centroid_as_start, use_bbox_center_as_start)

        # Handle the result of the route generation
        self.on_route_generated(message)

        # Prompt the user for the GPX output path
        # gpx_output_path = QFileDialog.getSaveFileName(self.detailsDlg, "Save Route as GPX", "", "*.gpx")[0]

        # Prepare the GPX output path by replacing the .pdf extension with .gpx
        gpx_output_path = output_path.replace('.pdf', '.gpx')

        # Check if the GPX output path is valid and if the route exists
        if gpx_output_path:
            if self.route:
                # Convert the route to GPX format and save it
                self.convert_route_to_gpx(self.route, gpx_output_path)
                # Initialize and assign the RouteAnimator
                self.detailsDlg.route_animator = RouteAnimator(self.iface, gpx_output_path)
            else:
                # Show an error message if no route data is available
                QMessageBox.critical(None, "Route Error", "No route data available to convert to GPX.")

    def extract_subcategory(self, row, subcategories):
        for sub in subcategories:
            for col in ['amenity', 'tourism', 'shop']:
                if col in row and pd.notna(row[col]) and sub in row[col]:
                    return sub
        return ''

    def greedy_tsp(self, points):
        # Number of points to visit
        n = len(points)

        # List to keep track of visited points
        visited = [False] * n

        # Start the tour from the first point
        tour = [0]
        visited[0] = True

        # Iterate over the number of points to be visited
        for _ in range(1, n):
            last = tour[-1]  # The last point in the current tour
            nearest = None  # Variable to store the nearest unvisited point
            # Initialize the nearest distance with infinity to ensure that any actual distance computed
            # in the subsequent iterations will be smaller than nearest_dist on the first comparison
            nearest_dist = float('inf')

            # Find the nearest unvisited point
            for i in range(n):
                if not visited[i]:
                    dist = geodesic(points[last], points[i]).meters
                    if dist < nearest_dist:
                        nearest = i
                        nearest_dist = dist

            # Add the nearest point to the tour and mark it as visited
            tour.append(nearest)
            visited[nearest] = True

        return tour  # Return the order of points in the tour

    def process_route_generation(self, category, output_path, address, eps, min_pts,
                                 use_cluster_centroid_as_start, use_bbox_center_as_start):
        # Get the place name from the GeoDataFrame
        place_name = self.gdf['display_name'].iloc[0]

        # Define categories and their respective subcategories
        categories = {
            'tourism': ['hotel', 'museum'],
            'amenity': ['restaurant', 'bar', 'cafe'],
            'shop': ['mall']
        }

        pois_list = []

        try:
            # Fetch POIs based on the selected category
            if category == "All":
                for main_category, subcategories in categories.items():
                    try:
                        pois = ox.features_from_place(place_name, tags={main_category: subcategories}).to_crs(epsg=4326)
                        if pois.empty:
                            continue
                        pois['category'] = main_category

                        pois['subcategory'] = pois.apply(lambda row: self.extract_subcategory(row, subcategories),
                                                         axis=1)
                        pois_list.append(pois)
                    except ValueError:
                        pass
            else:
                selected_subcategories = self.detailsDlg.get_selected_categories()
                for main_category, subcategories in categories.items():
                    relevant_subcategories = list(set(subcategories) & set(selected_subcategories))
                    if relevant_subcategories:
                        try:
                            pois = ox.features_from_place(place_name,
                                                          tags={main_category: relevant_subcategories}).to_crs(
                                epsg=4326)
                            if pois.empty:
                                continue
                            pois['category'] = main_category
                            pois['subcategory'] = pois.apply(lambda row: self.extract_subcategory(row, subcategories),
                                                             axis=1)
                            pois_list.append(pois)
                        except ValueError:
                            pass

            if not pois_list:
                raise ValueError("No data elements found for the given query.")

            # Concatenate all POIs into a single DataFrame
            pois = pd.concat(pois_list, ignore_index=True)
        except ValueError as e:
            QMessageBox.warning(self.detailsDlg, "Data Error", str(e))
            self.detailsDlg.generateRouteButton.setEnabled(True)
            return

        # Save the POIs to a layer in QGIS
        self.save_pois_layer(pois)
        waypoints = None
        # QgsMessageLog.logMessage(str(pois))

        # Run DBSCAN clustering on the POIs
        largest_cluster_centroid, densest_cluster_centroid, largest_cluster_sample = (
            self.cluster_analysis.run_dbscan_clustering(
                pois, place=place_name, eps=eps, minpts=min_pts))

        if largest_cluster_sample is None:
            QMessageBox.warning(self.detailsDlg, "Clustering Error", "No clusters found.")
            self.detailsDlg.generateRouteButton.setEnabled(True)
            return

        # QgsMessageLog.logMessage(str(densest_cluster_centroid))
        # Determine the starting point for the route
        if use_cluster_centroid_as_start:
            # starting_point = None
            starting_point = [[densest_cluster_centroid.x, densest_cluster_centroid.y]]
            if starting_point and isinstance(starting_point[0], list):
                starting_point = starting_point[0]
            starting_point = [starting_point[1], starting_point[0]]
        elif use_bbox_center_as_start:
            starting_point = (self.city_point.y, self.city_point.x)
        else:
            starting_point = self.geocode_address(address) if address else None
            if not starting_point:
                try:
                    starting_point = self.get_saved_point_coordinates()
                except ValueError as e:
                    QMessageBox.warning(self.detailsDlg, "Error", str(e))
                    self.detailsDlg.generateRouteButton.setEnabled(True)
                    return

        # Use a maximum number of POIs from the largest cluster as waypoints
        waypoints = [[point.x, point.y] for point in largest_cluster_sample.geometry]

        if not waypoints:
            QMessageBox.warning(self.detailsDlg, "Waypoint Error", "No valid waypoints found for the route.")
            self.detailsDlg.generateRouteButton.setEnabled(True)
            return

        # Include the starting point in the list of points for TSP
        points = [starting_point] + waypoints

        # Solve TSP to order the waypoints
        tour_indices = self.greedy_tsp(points)
        ordered_points = [points[i] for i in tour_indices]

        # Log the ordered waypoints
        # QgsMessageLog.logMessage(f"Ordered waypoints: {ordered_points}", "OptimalCustomRoute", Qgis.Info)

        # Create the route using the ordered points
        route = self.create_route(ors_client, ordered_points[0], ordered_points[1:])
        # route = self.create_route(ors_client, points[0], points[1:])
        # route = self.create_route(ors_client, starting_point, waypoints)
        self.route = route
        self.create_route_layer(route)
        self.export_pdf.export_to_pdf(output_path, self.route_layer)

        return "Route saved as PDF."

    def get_saved_point_coordinates(self):
        layers = QgsProject.instance().mapLayersByName(self.point_layer_name)
        if not layers:
            raise ValueError("Starting point layer not found.")

        layer = layers[0]
        features = list(layer.getFeatures())
        if not features:
            raise ValueError("No features found in the starting point layer.")

        feature = features[0]
        geom = feature.geometry().asPoint()
        return (geom.y(), geom.x())

    def geocode_address(self, address):
        try:
            params = {'text': address}
            result = ors_client.pelias_search(**params)
            if result['features']:
                location = result['features'][0]['geometry']['coordinates']
                return (location[1], location[0])
            else:
                QMessageBox.warning(self.detailsDlg, "Geocoding Error", "Could not geocode the address.")
                return None
        except Exception as e:
            QMessageBox.critical(self.detailsDlg, "Geocoding Error", f"An error occurred: {e}")
            return None

    def save_or_replace_point(self):
        if not self.saved_point:
            self.add_starting_point()
        else:
            self.replace_point()
            self.detailsDlg.addressInput.setDisabled(False)
            self.detailsDlg.saveStartingPointButton.setText("Save Starting Point")

    def add_starting_point(self):
        address = self.detailsDlg.addressInput.text().strip()
        if not address:
            QMessageBox.warning(self.detailsDlg, "Input Error", "Please enter an address.")
            return

        location = self.geocode_address(address)
        if not location:
            return

        point = QgsPointXY(location[1], location[0])
        self.add_point_layer(point)
        self.detailsDlg.addressInput.setDisabled(True)
        self.detailsDlg.addressInputDisabled = True
        self.saved_point = True
        self.detailsDlg.saveStartingPointButton.setText("Replace Point")

    def replace_point(self):
        existing_layers = QgsProject.instance().mapLayersByName(self.point_layer_name)
        if existing_layers:
            QgsProject.instance().removeMapLayer(existing_layers[0])
        self.add_starting_point()
        self.detailsDlg.addressInput.setDisabled(True)
        self.detailsDlg.addressInputDisabled = True

    def add_point_layer(self, point):
        layer = QgsVectorLayer("Point?crs=EPSG:4326", self.point_layer_name, "memory")
        provider = layer.dataProvider()
        provider.addAttributes([QgsField("name", QVariant.String)])
        layer.updateFields()

        feature = QgsFeature()
        feature.setGeometry(QgsGeometry.fromPointXY(point))
        feature.setAttributes([self.point_layer_name])
        provider.addFeature(feature)

        layer.updateExtents()
        QgsProject.instance().addMapLayer(layer)
        self.iface.mapCanvas().setExtent(layer.extent())
        self.iface.mapCanvas().refresh()

    def save_starting_point_layer(self, starting_point):
        layer = QgsVectorLayer("Point?crs=EPSG:4326", "Starting Point", "memory")
        provider = layer.dataProvider()
        provider.addAttributes([QgsField("name", QVariant.String)])
        layer.updateFields()

        feature = QgsFeature()
        feature.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(starting_point[1], starting_point[0])))
        feature.setAttributes(["Starting Point"])
        provider.addFeature(feature)

        QgsProject.instance().addMapLayer(layer)

    def save_pois_layer(self, pois):
        layer = QgsVectorLayer("Point?crs=EPSG:4326", "selected_pois", "memory")
        provider = layer.dataProvider()
        provider.addAttributes([
            QgsField("name", QVariant.String),
            QgsField("category", QVariant.String),
            QgsField("subcategory", QVariant.String)
        ])
        layer.updateFields()

        features = []
        for idx, poi in pois.iterrows():
            feature = QgsFeature()
            feature.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(poi.geometry.centroid.x, poi.geometry.centroid.y)))
            feature.setAttributes([poi.get('name', ''), poi.get('category', ''), poi.get('subcategory', '')])
            features.append(feature)

        provider.addFeatures(features)
        QgsProject.instance().addMapLayer(layer)
        self.iface.mapCanvas().refresh()  # Ensure canvas refresh

    def convert_route_to_gpx(self, route, gpx_path):
        if not route:
            QMessageBox.critical(self.detailsDlg,"Route Error", "No route data available to convert to GPX.")
            return

        gpx = gpxpy.gpx.GPX()

        # Create GPX track
        gpx_track = gpxpy.gpx.GPXTrack()
        gpx.tracks.append(gpx_track)

        # Create GPX track segment
        gpx_segment = gpxpy.gpx.GPXTrackSegment()
        gpx_track.segments.append(gpx_segment)

        # Add route points to the GPX segment
        coordinates = route['features'][0]['geometry']['coordinates']
        for coord in coordinates:
            gpx_segment.points.append(gpxpy.gpx.GPXTrackPoint(coord[1], coord[0]))

        # Write GPX to file
        with open(gpx_path, 'w') as gpx_file:
            gpx_file.write(gpx.to_xml())

        QMessageBox.information(self.detailsDlg, "Success", f"Route saved as GPX at {gpx_path}")

    def create_route(self, client, starting_point, waypoints):
        try:
            # Log the starting point if provided
            # if starting_point:
            #    QgsMessageLog.logMessage(f"Starting point: {starting_point}", "OptimalCustomRoute", Qgis.Info)
            # else:
            #    QgsMessageLog.logMessage("No starting point provided", "OptimalCustomRoute", Qgis.Info)

            # QgsMessageLog.logMessage(f"Waypoints: {waypoints}", "OptimalCustomRoute", Qgis.Info)

            # Initialize coordinates list with the starting point, if provided
            coordinates = [[starting_point[1], starting_point[0]]] if starting_point else []
            # QgsMessageLog.logMessage(f"Initial coordinates: {coordinates}", "OptimalCustomRoute", Qgis.Info)

            # Add waypoints to the coordinates list after validation
            for waypoint in waypoints:
                if isinstance(waypoint, list) and len(waypoint) == 2 and all(
                        isinstance(coord, (int, float)) for coord in waypoint):
                    # QgsMessageLog.logMessage(f"Adding waypoint: {waypoint}", "OptimalCustomRoute", Qgis.Info)
                    coordinates.append(waypoint)
                else:
                    raise ValueError(f"Invalid waypoint format: {waypoint}")

            # QgsMessageLog.logMessage(f"Extended coordinates: {coordinates}", "OptimalCustomRoute", Qgis.Info)

            # Ensure there are enough points to generate a route
            if len(coordinates) < 2:
                raise ValueError("Not enough points to generate a route")

            # Log the final coordinates being sent to the API
            # QgsMessageLog.logMessage(f"Final coordinates for routing: {coordinates}", "OptimalCustomRoute", Qgis.Info)

            # Request route from OpenRouteService API
            route = client.directions(
                coordinates=coordinates,
                preference='shortest',
                profile='driving-car',
                format='geojson',
            )

            return route
        except ApiError as e:
            # Handle API errors
            QgsMessageLog.logMessage(f"API Error: {e}", "OptimalCustomRoute", Qgis.Critical)
            QMessageBox.critical(self.detailsDlg, "Route Error", f"An error occurred while generating the route: {e}")
            return None
        except ValueError as e:
            # Handle value errors
            QgsMessageLog.logMessage(f"ValueError: {e}", "OptimalCustomRoute", Qgis.Critical)
            QMessageBox.critical(self.detailsDlg, "Route Error", f"An error occurred: {e}")
            return None
        except Exception as e:
            # Handle unexpected errors
            QgsMessageLog.logMessage(f"Unexpected Error: {e}", "OptimalCustomRoute", Qgis.Critical)
            QMessageBox.critical(self.detailsDlg, "Route Error", f"An unexpected error occurred: {e}")
            return None

    def create_route_layer(self, route):
        # Check if the route data is valid
        if not route:
            return

        try:
            # Extract coordinates from the route geometry
            coordinates = route['features'][0]['geometry']['coordinates']

            # Convert coordinates to QgsPointXY objects
            points = [QgsPointXY(coord[0], coord[1]) for coord in coordinates]

            # Create a new memory vector layer for the route
            self.route_layer = QgsVectorLayer("LineString?crs=EPSG:4326", "Route", "memory")
            provider = self.route_layer.dataProvider()

            # Add a "name" field to the layer
            provider.addAttributes([QgsField("name", QVariant.String)])
            self.route_layer.updateFields()

            # Create a new feature for the route
            route_feature = QgsFeature()
            route_geometry = QgsGeometry.fromPolylineXY(points)
            route_feature.setGeometry(route_geometry)
            route_feature.setAttributes(["Route"])
            # Add the feature to the layer
            provider.addFeature(route_feature)

            # Update layer extents and add it to the QGIS project
            self.route_layer.updateExtents()
            QgsProject.instance().addMapLayer(self.route_layer)

            # Customize the route layer's renderer
            renderer = self.route_layer.renderer()
            symbol = renderer.symbol()
            symbol.setWidth(1.2)
            self.route_layer.triggerRepaint()

            # Create a new memory vector layer for the start and end points
            points_layer = QgsVectorLayer("Point?crs=EPSG:4326", "Points", "memory")
            provider = points_layer.dataProvider()

            # Add a "name" field to the layer
            provider.addAttributes([QgsField("name", QVariant.String)])
            points_layer.updateFields()

            # Create and add a feature for the start point
            start_feature = QgsFeature()
            start_feature.setGeometry(QgsGeometry.fromPointXY(points[0]))
            start_feature.setAttributes(["Start"])
            provider.addFeature(start_feature)

            # Create and add a feature for the end point
            end_feature = QgsFeature()
            end_feature.setGeometry(QgsGeometry.fromPointXY(points[-1]))
            end_feature.setAttributes(["End"])
            provider.addFeature(end_feature)

            # Update layer extents and add it to the QGIS project
            points_layer.updateExtents()
            QgsProject.instance().addMapLayer(points_layer)

            # Customize the points layer's renderer
            start_symbol = QgsMarkerSymbol.createSimple({'name': 'circle', 'color': 'green', 'size': '5'})
            end_symbol = QgsMarkerSymbol.createSimple({'name': 'circle', 'color': 'red', 'size': '5'})
            points_renderer = QgsCategorizedSymbolRenderer("name", [
                QgsRendererCategory("Start", start_symbol, "Start"),
                QgsRendererCategory("End", end_symbol, "End")
            ])
            points_layer.setRenderer(points_renderer)
            points_layer.triggerRepaint()

            # Set the map canvas extent to the route layer's extent and refresh the canvas
            self.iface.mapCanvas().setExtent(self.route_layer.extent())
            self.iface.mapCanvas().refresh()
        except KeyError as e:
            # Log and show an error message if there is a KeyError
            QgsMessageLog.logMessage(f"KeyError: {e}", "OptimalCustomRoute", Qgis.Critical)
            QMessageBox.critical(self.detailsDlg, "Route Error", f"An error occurred while processing the route data: {e}")


    def on_route_generated(self, message):
        QMessageBox.information(self.detailsDlg, "Success", message)
        self.detailsDlg.generateRouteButton.setEnabled(True)


def main():
    OptimalCustomRoute(iface)


main()
