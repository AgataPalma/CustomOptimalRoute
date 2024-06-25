from PyQt5.QtWidgets import QMessageBox
from qgis.PyQt.QtCore import QThread, QTimer, pyqtSlot
from qgis.PyQt.QtGui import QColor
from qgis.gui import QgsVertexMarker, QgsMapCanvas
from qgis.core import QgsPointXY
import gpxpy
import gpxpy.gpx


class RouteAnimator(QThread):
    def __init__(self, iface, gpx_path):
        QThread.__init__(self)
        self.iface = iface
        self.gpx_path = gpx_path
        self.mainWindow = None
        self.actions = []
        self.first_start = True
        self.canvas = self.iface.mapCanvas() if self.iface else QgsMapCanvas()
        self.marker_list = []
        self.j = 0
        self.timer = QTimer(self.canvas)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.draw_track)
        self.load_gpx_data()


    def load_gpx_data(self):
        try:
            with open(self.gpx_path, 'r') as gpx_file:
                gpx = gpxpy.parse(gpx_file)
                for track in gpx.tracks:
                    for segment in track.segments:
                        for point in segment.points:
                            marker = QgsVertexMarker(self.canvas)
                            marker.setCenter(QgsPointXY(point.longitude, point.latitude))
                            marker.setColor(QColor(255, 0, 0))
                            marker.setFillColor(QColor(255, 255, 0))
                            marker.setIconSize(10)
                            marker.setIconType(QgsVertexMarker.ICON_BOX)
                            marker.setPenWidth(4)
                            self.marker_list.append(marker)
                self.n_track = len(self.marker_list)
        except Exception as e:
            QMessageBox.critical(None, "GPX Load Error", f"An error occurred while loading the GPX file: {e}")

    def hide_track(self):
        self.j = 0
        for marker in self.marker_list:
            marker.hide()

    @pyqtSlot()
    def draw_track(self):
        if self.j >= self.n_track:
            self.hide_track()
            self.j = 0
            self.quit()
        else:
            self.marker_list[self.j].show()
            self.j += 1

    def start_animation(self):
        self.hide_track()
        self.timer.start()

    def stop_animation(self):
        self.hide_track()
        self.timer.stop()

    def run(self):
        self.start_animation()

    def quit(self):
        self.hide_track()
        self.stop_animation()
