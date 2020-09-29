from typing import Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf

from src.simulation.simulation_view import SimulationView
from src.model.network_model import NetworkModel
from src.model.sac_training import SACTraining
from src.model.supervised_training import SupervisedTraining

import math
import src.simulation.utils as utils

Axes3D = Axes3D  # pycharm auto import

# reset console, just in case
tf.keras.backend.clear_session()

np.random.seed(1)
np.set_printoptions(precision=3, suppress=True)


def foodSliderToFood(value):
    food_set = 0
    if value >= 0:
        food_set = 10.0 ** (value * 6.0 + 1.0)
    return food_set


def foodToFoodSlider(value):
    food_set = -1e-3
    if value > 0.0:
        food_set = (math.log10(value) - 1.0)/6.0
    return food_set


class SimulationGUI:
    def __init__(self):
        # ///////////////
        #   GUI - INIT
        # ///////////////

        self.A_agreeable_tick_size = [1, 2, 4, 5, 10, 20, 25, 40, 50, 75]
        self.sv = SimulationView([8000, 2000], [0.0, 0.0])
        self.sm = NetworkModel(
        #    SupervisedTraining(self.sv.simulation.get_inversion()))
            SACTraining(self.sv))

        self.marker_index = -1

        self.plt_line_width = 1
        self.plt_mark_size = 0.8
        self.x_ticks_rotation = 45

        # Pycharmer
        self.first_plot: plt.Axes
        self.second_plot: plt.Axes
        self.figure: plt.Figure

        self.figure, (self.first_plot, self.second_plot) = plt.subplots(1, 2, figsize=(12, 8))
        self.figure.subplots_adjust(left=0.1, bottom=0.25, right=0.90, top=0.95)

        self.food_slider = None
        self.next_button = None
        self.interactive_plot()

        self.figure.canvas.mpl_connect('key_press_event', self.handle_pressed_keyboard)

    def interactive_plot(self):

        # //////////
        #   SLIDER
        # //////////

        food_axis = plt.axes([0.05, 0.10, 0.85, 0.03], facecolor='lightgreen')
        self.food_slider = Slider(food_axis, 'Food', -1e-3, 1.0, valinit=0.0, facecolor='green', dragging=True)

        self.setSlider(self.food_slider.valinit)
        self.food_slider.on_changed(self.updateSlider)

        # /////////
        #   BUTTON
        # /////////

        next_axis = plt.axes([0.8, 0.025, 0.1, 0.04])
        self.next_button = Button(next_axis, 'Next', color='powderblue', hovercolor='mediumturquoise')
        self.next_button.on_clicked(self.next_step)

    def plot(self):
        # //////////
        #   GUI - PLOTTING - X Sine
        # //////////

        self.first_plot.plot(self.sv.linear, self.sv.prey, 'o-', color='limegreen', linewidth=self.plt_line_width,
                             markersize=self.plt_mark_size)
        self.first_plot.plot(self.sv.linear, self.sv.predator, 'o-', color='red', linewidth=self.plt_line_width,
                             markersize=self.plt_mark_size)
        self.first_plot.plot(self.sv.linear, self.sv.food_decision, 'o-', color='darkgreen',
                             linewidth=self.plt_line_width,
                             markersize=self.plt_mark_size)
        self.first_plot.set_yscale("log")

        ticks_range = max(np.max(self.sv.prey), np.max(self.sv.predator), np.max(self.sv.food))
        self.first_plot.set_xticks(
            ticks=np.arange(0, self.sv.K + 1, step=utils.findFairTick(self.sv.K, 10, self.A_agreeable_tick_size)))
        self.first_plot.tick_params(axis='x', labelrotation=self.x_ticks_rotation)
        self.first_plot.set_yticks(
            ticks=np.arange(10, ticks_range + 1, step=utils.findFairTick(ticks_range, 15, self.A_agreeable_tick_size)))

        # //////////
        #   GUI - PLOTTING - XY Spiral
        # //////////

        if self.marker_index != -1:
            self.second_plot.plot(self.sv.prey[self.marker_index:self.marker_index + 1],
                                  self.sv.predator[self.marker_index:self.marker_index + 1], 'ko-',
                                  color='white')
        self.second_plot.plot(self.sv.prey, self.sv.predator, 'ko-', linewidth=self.plt_line_width,
                              markersize=self.plt_mark_size)
        self.marker_index = self.sv.K - 1
        self.second_plot.plot(self.sv.prey[self.marker_index:self.marker_index + 1],
                              self.sv.predator[self.marker_index:self.marker_index + 1], 'ko-',
                              color='orange')

        max_x_lv_prey = np.max(self.sv.prey) * 1.101
        max_y_lv_predator = np.max(self.sv.predator) * 1.101
        self.second_plot.set_xticks(
            ticks=np.arange(0, max_x_lv_prey, step=utils.findFairTick(max_x_lv_prey, 10, self.A_agreeable_tick_size)))
        self.second_plot.tick_params(axis='x', rotation=self.x_ticks_rotation)
        self.second_plot.set_yticks(
            ticks=np.arange(0, max_y_lv_predator, step=utils.findFairTick(max_y_lv_predator, 10, self.A_agreeable_tick_size)))

    def updateSlider(self, value):
        food_set = foodSliderToFood(value)
        self.food_slider.valtext.set_text(utils.significantFormat(food_set))
        self.figure.canvas.draw_idle()

    def setSlider(self, value):
        self.food_slider.set_val(value)

    def predict_step(self):
        obs, _ = self.sv.get_normalised() # TODO INCLUDE RESOURCES
        proposed_action = self.sm.act(obs)

        self.setSlider(proposed_action[0])

        return proposed_action

    def next_step(self, event, **kwargs):
        count = kwargs.get('count', 1)
        for i in range(count):
            food = foodSliderToFood(self.food_slider.val)
            reward, _ = self.sv.step([food, 0])

            print('REWARD', reward)

            # TODO NORMALISATION INSIDE SM.ACT SM.ACT INTO NEW METHOD SM.DECIDE???
            self.predict_step()

        self.plot()
        plt.draw()

    def _reset(self, function):
        self.marker_index = 0
        self.first_plot.cla()
        self.second_plot.cla()

        function()
        prediction = self.predict_step()
        self.plot()

    def reset(self):
        self._reset(self.sv.reset)

    def restart(self):
        self._reset(self.sv.restart)

    def handle_pressed_keyboard(self, event):
        # /////////
        #   KEYBOARD INPUT
        # /////////
        if event.key == 'n':
            self.next_step(event, count=1)
        if event.key == 'm':
            self.next_step(event, count=10)
        if event.key == ',':
            self.next_step(event, count=100)
        if event.key == 'r':
            self.reset()
        if event.key == 'k':
            self.restart()

    def run(self):
        self.reset()
        plt.show()


s_gui = SimulationGUI()
s_gui.run()
print("STOP")
