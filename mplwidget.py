from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('QT5Agg')
import warnings

class MplCanvas(Canvas):
    def __init__(self):
        plt.rcParams['axes.facecolor'] = 'black'  #grouping the axes then making their background color black
        plt.rc('axes', edgecolor='w')
        plt.rc('xtick', color='w')
        plt.rc('ytick', color='w')
        plt.rcParams['savefig.facecolor'] = 'black' #grouping the figures in the graph then making their background color black
        plt.rcParams["figure.autolayout"] = True
        self.figure = plt.figure()
        self.figure.patch.set_facecolor('black')
        self.figure.add_subplot()
        Canvas.__init__(self, self.figure)
        Canvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        Canvas.updateGeometry(self)

    def plot_spectrogram(self, signal, fs):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            self.figure.clear()
            axes = self.figure.add_subplot()
            spectrogram = axes.specgram(signal, Fs=fs, cmap='inferno')
            self.figure.colorbar(spectrogram[3], ax=axes)
            axes.set_ylim(0)
            self.draw()


class MplWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.canvas = MplCanvas()
        self.vbl = QtWidgets.QVBoxLayout()
        self.vbl.addWidget(self.canvas)
        self.setLayout(self.vbl)
