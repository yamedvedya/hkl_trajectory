# Created by matveyev at 15.02.2021

APP_NAME = "HKL_Viewer"

import os
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

        self.motors = [name.split('  ')[0] for name in PyTango.DeviceProxy(self.diffrac).motorlist]

        self.real_trajectory = None

        self.original_plots = {}
        self.reduce_plots = {}

        self.command = None
        self.starts = None
        self.stops = None
        self.nb_steps = None
        self.orig_steps = None
        self.integ_time = None

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

        self.sb_to_slider('hkl_grid')
        self.sb_to_slider('min_move')
        self.sb_to_slider('lin_threshold')

        for ui in ['omega_t', 'omega', 'phi', 'chi', 'mu', 'gamma', 'delta']:
            getattr(self._ui, f'chk_{ui}_lin').clicked.connect(self.calculate)
            getattr(self._ui, f'chk_{ui}_exclude').clicked.connect(self.calculate)

        self._ui.le_command.editingFinished.connect(self.new_command)

        self._ui.sl_hkl_grid.valueChanged.connect(lambda: self.slider_to_sb('hkl_grid'))
        self._ui.sl_min_move.valueChanged.connect(lambda: self.slider_to_sb('min_move'))
        self._ui.sl_lin_threshold.valueChanged.connect(lambda: self.slider_to_sb('lin_threshold'))

        self._ui.sb_hkl_grid.valueChanged.connect(lambda: self.sb_to_slider('hkl_grid'))
        self._ui.sb_min_move.valueChanged.connect(lambda: self.sb_to_slider('min_move'))
        self._ui.sb_lin_threshold.valueChanged.connect(lambda: self.sb_to_slider('lin_threshold'))

        self._ui.sl_round.valueChanged.connect(lambda: self.sync_round('sl'))
        self._ui.sb_round.valueChanged.connect(lambda: self.sync_round('sb'))

        self._ui.cmd_calculate.clicked.connect(self.start)
        self._ui.cmd_save_script.clicked.connect(self.save_script)
        self._ui.cmd_load_script.clicked.connect(self.load_script)

        self.reload_real = True

    # ----------------------------------------------------------------------
    def block_signals(self, flag):
        self._ui.le_command.blockSignals(flag)

        self._ui.sl_hkl_grid.blockSignals(flag)
        self._ui.sl_min_move.blockSignals(flag)
        self._ui.sl_lin_threshold.blockSignals(flag)

        self._ui.sb_hkl_grid.blockSignals(flag)
        self._ui.sb_min_move.blockSignals(flag)
        self._ui.sb_lin_threshold.blockSignals(flag)

        self._ui.sl_round.blockSignals(flag)
        self._ui.sb_round.blockSignals(flag)

        for name in self.motors:
            getattr(self._ui, f'chk_{name}_exclude').blockSignals(flag)
            getattr(self._ui, f'chk_{name}_lin').blockSignals(flag)

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
    def sb_to_slider(self, name):
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
    def parse_command(self):
        self.command = None
        self.starts = None
        self.stops = None
        self.nb_steps = None
        self.orig_steps = None
        self.integ_time = None

        command_tokens = self._ui.le_command.text().split()
        if not command_tokens:
            self.report_error('Empty command')
        if command_tokens[0].strip('scan') in ['hkl', 'hkld', 'hklc', 'hkldc'] and len(command_tokens) < 9:
            self.report_error('Not enough input!')
        elif len(command_tokens) < 5:
            self.report_error('Not enough input!')

        self.command = command_tokens[0].strip('scan').strip('c')

        self.starts = []
        self.stops = []

        if 'h' in self.command:
            self.starts.append(float(command_tokens[1]))
            self.stops.append(float(command_tokens[2]))
        elif 'k' in self.command:
            if 'hkl' in self.command:
                start_ind = 3
                stop_ind = 4
            else:
                start_ind = 1
                stop_ind = 2
            self.starts.append(float(command_tokens[start_ind]))
            self.stops.append(float(command_tokens[stop_ind]))
        elif 'h' in self.command:
            if 'hkl' in self.command:
                start_ind = 5
                stop_ind = 6
            else:
                start_ind = 1
                stop_ind = 2
            self.starts.append(float(command_tokens[start_ind]))
            self.stops.append(float(command_tokens[stop_ind]))

        self.integ_time = float(command_tokens[-1])
        self.orig_steps = int(command_tokens[-2])

    # ----------------------------------------------------------------------
    def get_real_coordinates(self):

        self.parse_command()

        hkl_grid = self._ui.sb_hkl_grid.value()

        self.nb_steps = self.orig_steps
        for start, stop in zip(self.starts, self.stops):
            self.nb_steps = max(self.nb_steps, int(np.ceil(np.abs(stop-start)/hkl_grid)) + 1)

        positions = {}
        for name, address in self.reciprocal_motors.items():
            positions[name] = PyTango.DeviceProxy(address).position

        hkl_trajectory = np.ones((self.nb_steps, 3))
        hkl_trajectory[:, 0] = positions['h']
        hkl_trajectory[:, 1] = positions['k']
        hkl_trajectory[:, 2] = positions['l']

        for ind, (start, stop) in enumerate(zip(self.starts, self.stops)):
            trajectory = np.linspace(start, stop, self.nb_steps)
            if 'd' in self.command:
                hkl_trajectory[:, ind] += trajectory
            else:
                hkl_trajectory[:, ind] = trajectory

        for ind, name in enumerate(['h', 'k', 'l']):
            self.original_plots[name].setData(np.arange(len(hkl_trajectory[:, ind])), hkl_trajectory[:, ind])

        self.real_trajectory = np.zeros((len(self.motors), self.nb_steps))
        for step, (h, k, l) in enumerate(hkl_trajectory):
            self.real_trajectory[:, step] = self.get_positions_for_hkl([h, k, l])

        for name, trajectory in zip(self.motors, self.real_trajectory):
            self.original_plots[name].setData(np.arange(len(trajectory)), trajectory)

        self.reload_real = False

        self.calculate()

    # ----------------------------------------------------------------------
    def calculate(self):

        if self.real_trajectory is None:
            return

        trajectories = []
        linear_move = [True for _ in self.motors]
        motors_with_movement = [False for _ in self.motors]

        for ind, (name, trajectory) in enumerate(zip(self.motors, self.real_trajectory)):
            displacement = trajectory - trajectory[0]

            reduced_trajectory = np.vstack((np.array([0,               trajectory[0]]),
                                            np.array([len(trajectory), trajectory[0]])))

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
    def save_script(self):
        file_name = QtWidgets.QFileDialog.getSaveFileName(self, 'File name',
                                                          '/home/p23user/sardanaMacros/scripts/',
                                                          "Script Files (*.scr)")

        if file_name[0]:
            f_name = file_name[0]
            if not f_name.endswith('.scr'):
                f_name += '.scr'

            self.parse_command()
            with open(f_name, 'w') as f_out:
                f_out.write(self.command + '\n')
                f_out.write(';'.join([str(val) for val in self.starts]) + '\n')
                f_out.write(';'.join([str(val) for val in self.stops])+ '\n')
                f_out.write(str(self.orig_steps) + '\n')
                f_out.write(str(self.integ_time) + '\n')

                f_out.write(';'.join([str(self._ui.sb_hkl_grid.value()),
                                      str(self._ui.sb_min_move.value()),
                                      str(self._ui.sb_lin_threshold.value()),
                                      str(self._ui.sb_round.value())]) + '\n')

                motor_cmd = ''
                for name in self.motors:
                    if getattr(self._ui, f'chk_{name}_exclude').isChecked():
                        motor_cmd += f'{name}:exclude;'
                    elif getattr(self._ui, f'chk_{name}_lin').isChecked():
                        motor_cmd += f'{name}:linear;'
                    else:
                        motor_cmd += f'{name}:free;'

                f_out.write(motor_cmd.strip(';') + '\n')

    # ----------------------------------------------------------------------
    def load_script(self):
        file_name = QtWidgets.QFileDialog.getOpenFileName(self, 'Script File', '/home/p23user/sardanaMacros/script',
                                                          "Script Files (*.scr)")

        if file_name[0]:
            self.block_signals(True)
            with open(file_name[0], 'r') as f_in:
                cmd = f_in.readline().strip('\n') + 'cscan '

                starts = np.array([float(pos) for pos in f_in.readline().strip('\n').split(';')], dtype='d')
                stops = np.array([float(pos) for pos in f_in.readline().strip('\n').split(';')], dtype='d')
                for start, stop in zip(starts, stops):
                    cmd += f'{start} {stop} '

                cmd += f_in.readline().strip('\n') + ' '
                cmd += f_in.readline().strip('\n')

                self._ui.le_command.setText(cmd)

                constants = f_in.readline().strip('\n').split(';')

                self._ui.sb_hkl_grid.setValue(float(constants[0]))
                self._ui.sb_min_move.setValue(float(constants[1]))
                self._ui.sb_lin_threshold.setValue(float(constants[2]))

                self.sb_to_slider('hkl_grid')
                self.sb_to_slider('min_move')
                self.sb_to_slider('lin_threshold')

                self._ui.sl_round.setValue(float(constants[3]))
                self._ui.sl_round.setValue(float(constants[3]))

                states = f_in.readline().strip('\n').split(';')
                for name, state in zip(self.motors, states):
                    getattr(self._ui, f'chk_{name}_exclude').setChecked(False)
                    getattr(self._ui, f'chk_{name}_lin').setChecked(False)
                    if state == 'exclude':
                        getattr(self._ui, f'chk_{name}_exclude').setChecked(True)
                    elif state == 'linear':
                        getattr(self._ui, f'chk_{name}_lin').setChecked(True)

            self.block_signals(False)
            # self.get_real_coordinates()

    # ----------------------------------------------------------------------
    def report_error(self, text, informative_text='', detailed_text=''):

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
    def closeEvent(self, event):
        """
        """
        self._save_ui_settings()
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