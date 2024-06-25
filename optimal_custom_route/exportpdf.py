from qgis.core import QgsProject, QgsPrintLayout, QgsLayoutSize, QgsUnitTypes, QgsLayoutItemMap, QgsRectangle, \
    QgsLayoutPoint, QgsLayoutExporter
from datetime import datetime


class PdfExport:
    def __init__(self, canvas):
        self.canvas = canvas

    #
    def export_to_pdf(self, pdf_path, route_layer):
        # Get the current QGIS project instance
        project = QgsProject.instance()
        # Retrieve the layout manager from the project
        manager = project.layoutManager()
        # Create a unique layout name based on the current date and time
        layout_name = f"Route_Map_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        # Initialize a new print layout for the project
        layout = QgsPrintLayout(project)
        layout.initializeDefaults()
        layout.setName(layout_name)
        # Add the new layout to the layout manager
        manager.addLayout(layout)

        # Define A4 paper dimensions in millimeters
        a4_width = 210
        a4_height = 297
        # Get the first (and only) page in the layout
        layout_page = layout.pageCollection().pages()[0]
        # Set the page size to A4
        layout_page.setPageSize(
            QgsLayoutSize(a4_width, a4_height, QgsUnitTypes.LayoutMillimeters))  # A4 size in millimeters

        # Get the current map extent from the canvas
        map_extent = self.canvas.extent()
        map_width = map_extent.width()
        map_height = map_extent.height()

        # Adjust the page orientation if the map is wider than it is tall
        if map_width > map_height:
            a4_width, a4_height = a4_height, a4_width  #
            layout_page.setPageSize(
                QgsLayoutSize(a4_width, a4_height, QgsUnitTypes.LayoutMillimeters))

        # Create a map layout item and set its dimensions
        map = QgsLayoutItemMap(layout)
        map.setRect(0, 0, a4_width, a4_height)

        # Add a 10% buffer around the map extent to have a full view of the map
        buffer = 0.1  # 10% buffer
        new_extent = QgsRectangle(
            map_extent.xMinimum() - map_width * buffer,
            map_extent.yMinimum() - map_height * buffer,
            map_extent.xMaximum() + map_width * buffer,
            map_extent.yMaximum() + map_height * buffer
        )
        # Set the map item's extent to the new buffered extent
        map.setExtent(new_extent)
        # Enable a frame around the map item
        map.setFrameEnabled(True)
        # Zoom to the new extent if a route layer is provided
        if route_layer:
            map.zoomToExtent(new_extent)

        # Add the map item to the layout
        layout.addLayoutItem(map)

        # Position and size the map item within the layout
        map.attemptMove(QgsLayoutPoint(0, 0, QgsUnitTypes.LayoutMillimeters))
        map.attemptResize(QgsLayoutSize(a4_width, a4_height, QgsUnitTypes.LayoutMillimeters))

        # Export the layout to a PDF file
        exporter = QgsLayoutExporter(layout)
        result = exporter.exportToPdf(pdf_path, QgsLayoutExporter.PdfExportSettings())

        # Raise an exception if the export fails
        if result != QgsLayoutExporter.Success:
            raise Exception("Failed to export the layout to PDF")