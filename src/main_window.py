# Created by matveyev at 15.02.2021

APP_NAME = "HKL_Viewer"

import os
import logging
import psutil
import PyTango

import numpy as np
import pyqtgraph as pg

from PyQt5 import QtWidgets, QtCore
from src.gui.main_window_ui import Ui_MainWindow


class HKLViewer(QtWidgets.QMainWindow):
    """
    """

    STATUS_TICK = 2000

    ORIGINAL_STYLE = pg.mkPen(color=(255, 20, 20), style=QtCore.Qt.DotLine, width=1)
    REDUCED_STYLE = pg.mkPen(color=(20, 20, 20), style=QtCore.Qt.SolidLine, width=2)

    LB_ERROR = "QLabel {color: rgb(255, 0, 0);}"
    LB_NORMAL = "QLabel {color: rgb(0, 0, 0);}"

    # ----------------------------------------------------------------------
    def __init__(self, options):
        """
        """
        super(HKLViewer, self).__init__()
        self._ui = Ui_MainWindow()
        self._ui.setupUi(self)

        self._load_ui_settings()

        self.diffrac = options.diff

        self.reciprocal_motors = {}
        for motor in PyTango.DeviceProxy(self.diffrac).hklpseudomotorlist:
            name, address = motor.split('  ')
            self.reciprocal_motors[name.split('_')[1]] = address.strip('(tango://').strip(')')

        self.motors = None
        self.real_trajectory = None

        self.original_plots = {}
        self.reduce_plots = {}

        self.labels = {}
        for plot in ['h', 'k', 'l', 'omega_t', 'omega', 'phi', 'chi', 'mu', 'gamma', 'delta']:
            plot_item = pg.PlotItem()
            plot_item.setMenuEnabled(False)
            plot_item.getViewBox().setMouseMode(pg.ViewBox.RectMode)

            self.original_plots[plot] = plot_item.plot([], [], pen=self.ORIGINAL_STYLE, name='Original')
            self.reduce_plots[plot] = plot_item.plot([], [], pen=self.REDUCED_STYLE, name='Reduced')

            getattr(self._ui, f'gv_{plot}').setStyleSheet("")
            getattr(self._ui, f'gv_{plot}').setBackground('w')
            getattr(self._ui, f'gv_{plot}').setObjectName(f'gv_{plot}')

            getattr(self._ui, f'gv_{plot}').setCentralItem(plot_item)
            getattr(self._ui, f'gv_{plot}').setRenderHints(getattr(self._ui, f'gv_{plot}').renderHints())

        self.sync_slider('hkl_grid')
        self.sync_slider('min_move')
        self.sync_slider('lin_threshold')

        for ui in ['omega_t', 'omega', 'phi', 'chi', 'mu', 'gamma', 'delta']:
            getattr(self._ui, f'chk_{ui}_lin').clicked.connect(self.calculate)
            getattr(self._ui, f'chk_{ui}_exclude').clicked.connect(self.calculate)

        self._ui.le_command.editingFinished.connect(self.new_command)

        self._ui.sl_hkl_grid.valueChanged.connect(lambda: self.slider_to_sb('hkl_grid'))
        self._ui.sl_min_move.valueChanged.connect(lambda: self.slider_to_sb('min_move'))
        self._ui.sl_lin_threshold.valueChanged.connect(lambda: self.slider_to_sb('lin_threshold'))

        self._ui.sb_hkl_grid.valueChanged.connect(lambda: self.sync_slider('hkl_grid'))
        self._ui.sb_min_move.valueChanged.connect(lambda: self.sync_slider('min_move'))
        self._ui.sb_lin_threshold.valueChanged.connect(lambda: self.sync_slider('lin_threshold'))

        self._ui.sl_round.valueChanged.connect(lambda: self.sync_round('sl'))
        self._ui.sb_round.valueChanged.connect(lambda: self.sync_round('sb'))

        self._ui.cmd_calculate.clicked.connect(self.start)

        self.reload_real = True

    # ----------------------------------------------------------------------
    def block_signals(self, flag):
        self._ui.sl_hkl_grid.blockSignals(flag)
        self._ui.sl_min_move.blockSignals(flag)
        self._ui.sl_lin_threshold.blockSignals(flag)

        self._ui.sb_hkl_grid.blockSignals(flag)
        self._ui.sb_min_move.blockSignals(flag)
        self._ui.sb_lin_threshold.blockSignals(flag)

        self._ui.sl_round.blockSignals(flag)
        self._ui.sb_round.blockSignals(flag)

    # ----------------------------------------------------------------------
    def sync_round(self, source):
        self.block_signals(True)
        if source == 'sl':
            self._ui.sb_round.setValue(self._ui.sl_round.value())
        else:
            self._ui.sl_round.setValue(self._ui.sb_round.value())
        self.block_signals(False)

    # ----------------------------------------------------------------------
    def slider_to_sb(self, name):
        self.block_signals(True)
        min = np.log(getattr(self._ui, f'sb_{name}').minimum())
        max = np.log(getattr(self._ui, f'sb_{name}').maximum())
        getattr(self._ui, f'sb_{name}').setValue(np.exp(min + (max-min)*getattr(self._ui, f'sl_{name}').value()/100))
        if name == 'hkl_grid':
            self.reload_real = True
        self.block_signals(False)

    # ----------------------------------------------------------------------
    def sync_slider(self, name):
        self.block_signals(True)
        value = np.log(getattr(self._ui, f'sb_{name}').value())
        min = np.log(getattr(self._ui, f'sb_{name}').minimum())
        max = np.log(getattr(self._ui, f'sb_{name}').maximum())
        getattr(self._ui, f'sl_{name}').setValue(100*(value-min)/(max-min))
        if name == 'hkl_grid':
            self.reload_real = True
        self.block_signals(False)

    # ----------------------------------------------------------------------
    def new_command(self):
        self.reload_real = True

    # ----------------------------------------------------------------------
    def start(self):
        if self.reload_real:
            self.get_real_coordinates()
        else:
            self.calculate()

    # ----------------------------------------------------------------------
    def get_positions_for_hkl(self, hkl_values):

        PyTango.DeviceProxy(self.diffrac).write_attribute("computetrajectoriessim", hkl_values)
        return PyTango.DeviceProxy(self.diffrac).trajectorylist[0]

    # ----------------------------------------------------------------------
    def get_hkl_for_positions(self, position):

        PyTango.DeviceProxy(self.diffrac).write_attribute("computehkl", position)
        return PyTango.DeviceProxy(self.diffrac).computehkl

    # ----------------------------------------------------------------------
    def get_real_coordinates(self):

        command_tokens = self._ui.le_command.text().split()
        if not command_tokens:
            self.report_error('Empty command')
        if command_tokens[0].strip('scan') in ['hkl', 'hkld', 'hklc', 'hkldc'] and len(command_tokens) < 9:
            self.report_error('Not enough input!')
        elif len(command_tokens) < 5:
            self.report_error('Not enough input!')

        hkl_grid = self._ui.sb_hkl_grid.value()

        command = command_tokens[0].strip('scan').strip('c')
        integration = float(command_tokens[-1])

        n_steps = int(command_tokens[-2])
        if 'hkl' in command:
            for ind in range(3):
                start = float(command_tokens[ind*2 + 1])
                stop = float(command_tokens[ind*2 + 2])
                n_steps = max(n_steps, int(np.ceil(np.abs(stop-start)/hkl_grid)) + 1)
        else:
            n_steps = max(n_steps, int(np.ceil(np.abs(float(command_tokens[2]) - float(command_tokens[1]))/hkl_grid))+1)

        positions = {}
        for name, address in self.reciprocal_motors.items():
            positions[name] = PyTango.DeviceProxy(address).position

        hkl_trajectory = np.ones((n_steps, 3))
        hkl_trajectory[:, 0] = positions['h']
        hkl_trajectory[:, 1] = positions['k']
        hkl_trajectory[:, 2] = positions['l']

        if 'h' in command:
            hkl_trajectory[:, 0] = np.linspace(float(command_tokens[1]), float(command_tokens[2]), n_steps)
            if 'd' in command:
                hkl_trajectory[:, 0] += positions['h']
        elif 'k' in command:
            if 'hkl' in command:
                hkl_trajectory[:, 1] = np.linspace(float(command_tokens[3]), float(command_tokens[4]), n_steps)
            else:
                hkl_trajectory[:, 1] = np.linspace(float(command_tokens[1]), float(command_tokens[2]), n_steps)
            if 'd' in command:
                hkl_trajectory[:, 1] += positions['k']
        elif 'h' in command:
            if 'hkl' in command:
                hkl_trajectory[:, 2] = np.linspace(float(command_tokens[5]), float(command_tokens[6]), n_steps)
            else:
                hkl_trajectory[:, 2] = np.linspace(float(command_tokens[1]), float(command_tokens[2]), n_steps)
            if 'd' in command:
                hkl_trajectory[:, 2] += positions['l']

        for ind, name in enumerate(['h', 'k', 'l']):
            self.original_plots[name].setData(np.arange(len(hkl_trajectory[:, ind])), hkl_trajectory[:, ind])

        self.motors = [name.split('  ')[0] for name in PyTango.DeviceProxy(self.diffrac).motorlist]
        #TODO TEMP
        self.motors = ['omega_t', 'mu', 'omega', 'chi', 'phi', 'gamma', 'delta']

        self.real_trajectory = np.zeros((len(self.motors), n_steps))
        for step, (h, k, l) in enumerate(hkl_trajectory):
            self.real_trajectory[:, step] = self.get_positions_for_hkl([h, k, l])

        for name, trajectory in zip(self.motors, self.real_trajectory):
            self.original_plots[name].setData(np.arange(len(trajectory)), trajectory)

        self.reload_real = False

        self.calculate()

    # ----------------------------------------------------------------------
    def calculate(self):

        if self.motors is None and self.real_trajectory is None:
            return

        trajectories = []
        linear_move = [True for _ in self.motors]
        motors_with_movement = [False for _ in self.motors]

        for ind, (name, trajectory) in enumerate(zip(self.motors, self.real_trajectory)):
            displacement = trajectory - trajectory[0]

            reduced_trajectory = np.vstack((np.array([0,               trajectory[0]]),
                                            np.array([len(trajectory), trajectory[0]])))

            # check if there is a movement:
            if np.any(np.abs(displacement) > self._ui.sb_min_move.value()):
                motors_with_movement[ind] = True

                if not getattr(self._ui, f'chk_{name}_exclude').isChecked():
                    lin_move = trajectory[0] + displacement[-1]/(len(trajectory)-1) * np.arange(len(trajectory))
                    reduced_trajectory = np.vstack((np.array([0,               trajectory[0]]),
                                                    np.array([len(trajectory), trajectory[-1]])))

                    if np.any(np.abs(trajectory - lin_move) > self._ui.sb_lin_threshold.value()):
                        linear_move[ind] = False
                        if not getattr(self._ui, f'chk_{name}_lin').isChecked():
                            displacement = np.round(displacement, self._ui.sb_round.value())

                            reduced_trajectory = np.array([0, trajectory[0]])
                            old_value = displacement[0]
                            for idx, value in enumerate(displacement):
                                if value != old_value:
                                    old_value = value
                                    reduced_trajectory = np.vstack((reduced_trajectory, np.array([idx, trajectory[idx]])))

            trajectories.append(reduced_trajectory)

        for name, trajectory, linear, moving in zip(self.motors, trajectories, linear_move, motors_with_movement):
            self.reduce_plots[name].setData(trajectory[:, 0], trajectory[:, 1])
            getattr(self._ui, f'lb_{name}_state').setStyleSheet(self.LB_NORMAL)
            if not moving:
                getattr(self._ui, f'lb_{name}_state').setText('No move')
                getattr(self._ui, f'chk_{name}_exclude').setEnabled(False)
                getattr(self._ui, f'chk_{name}_lin').setEnabled(False)
            else:
                if linear:
                    getattr(self._ui, f'lb_{name}_state').setText('Linear move')
                    getattr(self._ui, f'chk_{name}_exclude').setEnabled(True)
                    getattr(self._ui, f'chk_{name}_lin').setEnabled(False)
                else:
                    if self.check_movement_direction(trajectory):
                        getattr(self._ui, f'lb_{name}_state').setText('')
                    else:
                        getattr(self._ui, f'lb_{name}_state').setText('MOTION CANNOT BE EXECUTED!')
                        getattr(self._ui, f'lb_{name}_state').setStyleSheet(self.LB_ERROR)
                    getattr(self._ui, f'chk_{name}_exclude').setEnabled(True)
                    getattr(self._ui, f'chk_{name}_lin').setEnabled(True)

        reduced_hkl = np.zeros((3, self.real_trajectory.shape[1]))
        for point in range(reduced_hkl.shape[1]):
            motor_positions = np.zeros(len(self.motors))
            for ind, trajectory in enumerate(trajectories):
                motor_positions[ind] = np.interp(point, trajectory[:, 0], trajectory[:, 1])
            reduced_hkl[:, point] = self.get_hkl_for_positions(motor_positions)

        for ind, name in enumerate(['h', 'k', 'l']):
            self.reduce_plots[name].setData(np.arange(len(reduced_hkl[ind])), reduced_hkl[ind])

    @staticmethod
    # ----------------------------------------------------------------------
    def check_movement_direction(trajectory):
        break_points = []
        trajectory = np.transpose(trajectory)

        for row in np.diff(trajectory):
            if np.any(row < 0) and np.any(row > 0):
                sign = np.sign(row)

                start_ind = 0
                finished = False
                while not finished:
                    res = np.where(sign[start_ind:] != sign[start_ind])[0]
                    if len(res):
                        start_ind = start_ind + res[0]
                        break_points.append(start_ind)
                    else:
                        finished = True

        if break_points:
            return False
        else:
            return True

    # ----------------------------------------------------------------------
    def report_error(self, text, informative_text='', detailed_text=''):

        self.log.error("Error: {}, {}, {} ".format(text, informative_text, detailed_text))

        self.msg = QtWidgets.QMessageBox()
        self.msg.setModal(False)
        self.msg.setIcon(QtWidgets.QMessageBox.Critical)
        self.msg.setText(text)
        self.msg.setInformativeText(informative_text)
        if detailed_text != '':
            self.msg.setDetailedText(detailed_text)
        self.msg.setWindowTitle("Error")
        self.msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        self.msg.show()

    # ----------------------------------------------------------------------
    def _close_me(self):
        self.log.info("Closing the app...")
        self._save_ui_settings()

    # ----------------------------------------------------------------------
    def _exit(self):
        self._close_me()
        QtWidgets.QApplication.quit()

    # ----------------------------------------------------------------------
    def closeEvent(self, event):
        """
        """
        self._close_me()
        event.accept()

    # ----------------------------------------------------------------------
    def _save_ui_settings(self):
        """Save basic GUI settings.
        """
        settings = QtCore.QSettings(APP_NAME)

        settings.setValue("MainWindow/geometry", self.saveGeometry())
        settings.setValue("MainWindow/state", self.saveState())

    # ----------------------------------------------------------------------
    def _load_ui_settings(self):
        """Load basic GUI settings.
        """
        settings = QtCore.QSettings(APP_NAME)

        try:
            self.restoreGeometry(settings.value("MainWindow/geometry"))
        except:
            pass

        try:
            self.restoreState(settings.value("MainWindow/state"))
        except:
            pass