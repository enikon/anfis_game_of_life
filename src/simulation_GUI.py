from typing import Any, Tuple
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from stable_baselines3 import PPO

import utils
from distillation.model import load_trained_model
from simulation_view import SimulationView
from simulation_model import SimulationModel


def foodSliderToFood(value):
    food_set = 0
    if value >= 1.0:
        food_set = utils.float2log10int(value)
    return food_set


class SimulationGUI:
    def __init__(self):
        # ///////////////
        #   GUI - INIT
        # ///////////////

        self.A_agreeable_tick_size = [1, 2, 4, 5, 10, 20, 25, 40, 50, 75]
        self.sv = SimulationView([5000, 10000, 12000], [0.0, 0.0])
        # self.sm = SimulationModel(self.sv.simulation.get_inversion())
        # self.sm = SimulationModel(load_trained_model())

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
        self.food_slider = Slider(food_axis, 'Food', 0.999, 7.0, valinit=0.0, facecolor='green', dragging=True)

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
        self.first_plot.plot(self.sv.linear, self.sv.plant, 'o-', color='darkgreen', linewidth=self.plt_line_width,
                             markersize=self.plt_mark_size)
        self.first_plot.plot(self.sv.linear, self.sv.water_decision, 'o-', color='blue',
                             linewidth=self.plt_line_width,
                             markersize=self.plt_mark_size)
        self.first_plot.set_yscale("log")

        ticks_range = max(np.max(self.sv.prey), np.max(self.sv.predator), np.max(self.sv.water))
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
        obs = self.sv.get()
        obs_scaled = [((math.log(o, 10) - 1) / 6 if o > 0 else 0) for o in obs]
        action_scaled = self.sm.act(obs_scaled)  # TODO NORMALISATION AS FUNCTION
        action_slider = [a * 6 + 1 for a in action_scaled]
        self.setSlider(action_slider[0])

        return [10**action_slider_i for action_slider_i in action_slider]

    def next_step(self, event, **kwargs):
        count = kwargs.get('count', 1)
        for i in range(count):
            # TODO MAKE MORE PRIVATE FIELDS/METHODS IN CLASSES

            # TODO GUI VALUE 1.0-7.0
            # TODO GUI->INTERF 10K
            # TODO GUI->SIM 10K
            # TODO GUI->MODEL 0-1
            food = foodSliderToFood(self.food_slider.val)  # TODO NORMALISATION AS FUNCTION
            self.sv.step([food, 0])

            # TODO NORMALISATION INSIDE SM.ACT SM.ACT INTO NEW METHOD SM.DECIDE???
            # self.predict_step()

        self.plot()
        plt.draw()

    def reset(self, entities=None, resources=None):
        self.marker_index = 0
        self.first_plot.cla()
        self.second_plot.cla()

        self.sv.reset(entities, resources)
        # prediction = self.predict_step()
        # self.sv.supply([prediction[0], 0])
        self.sv.supply([0, 0])
        self.sv.collect()
        self.plot()

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
            a = np.random.uniform(2.0, 5.0)
            b = np.random.uniform(1.0, a - 0.3)
            self.reset([10 ** a, 10 ** b], [0, 0])

    def run(self):
        self.reset([5000, 10000, 12000], [0, 0])
        plt.show()


s_gui = SimulationGUI()
s_gui.run()
print("STOP")
