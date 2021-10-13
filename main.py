# ----------------------------------------------------------------------
# Author:        yury.matveev@desy.de
# ----------------------------------------------------------------------


"""
"""

import sys
from optparse import OptionParser

from PyQt5 import QtWidgets

from src.main_window import HKLViewer


# ----------------------------------------------------------------------
def main():
    parser = OptionParser()

    parser.add_option("-s", "--server", dest="diff", default='hasep23oh:10000/controller/diffrac6cp23/e6c',
                      help="DiffracDevice")

    (options, _) = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)

    mainWindow = HKLViewer(options)
    mainWindow.show()

    return app.exec_()


# ----------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
