from typing import Any, Tuple
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

import utils
from simulation_view import SimulationView
from simulation_model import SimulationModel

# ///////////////
#   GUI - INIT
# ///////////////

A_agreeable_tick_size = [1, 2, 4, 5, 10, 20, 25, 40, 50, 75]
sv = SimulationView()
sm = SimulationModel(sv.simulation.get_inversion())

plt_line_width = 1
plt_mark_size = 0.8
x_ticks_rotation = 45

marker_index = -1

#Pycharmer
first_plot: plt.Axes
second_plot: plt.Axes
figure: plt.Figure

figure, (first_plot, second_plot) = plt.subplots(1, 2, figsize=(12, 8))
figure.subplots_adjust(left=0.1, bottom=0.25, right=0.90, top=0.95)


def plot():
    global marker_index
    # //////////
    #   GUI - PLOTTING - X Sine
    # //////////

    first_plot.plot(sv.linear, sv.prey, 'o-', color='green', linewidth=plt_line_width, markersize=plt_mark_size)
    first_plot.plot(sv.linear, sv.predator, 'o-', color='red', linewidth=plt_line_width, markersize=plt_mark_size)
    first_plot.set_yscale("log")

    ticks_range = max(np.max(sv.prey), np.max(sv.predator))
    first_plot.set_xticks(ticks=np.arange(0, sv.K + 1, step=utils.findFairTick(sv.K, 10, A_agreeable_tick_size)))
    first_plot.tick_params(axis='x', labelrotation=x_ticks_rotation)

    first_plot.set_yticks(ticks=np.arange(10, ticks_range + 1, step=utils.findFairTick(ticks_range, 15, A_agreeable_tick_size)))

    # //////////
    #   GUI - PLOTTING - XY Spiral
    # //////////

    if marker_index != -1:
        second_plot.plot(sv.prey[marker_index:marker_index+1], sv.predator[marker_index:marker_index+1], 'ko-', color='white')
    second_plot.plot(sv.prey, sv.predator, 'ko-', linewidth=plt_line_width, markersize=plt_mark_size)
    marker_index = sv.K-1
    second_plot.plot(sv.prey[marker_index:marker_index+1],   sv.predator[marker_index:marker_index+1],   'ko-', color='orange')

    max_x_lv_prey = np.max(sv.prey)*1.1
    max_y_lv_predator = np.max(sv.predator)*1.1
    second_plot.set_xticks(ticks=np.arange(0, max_x_lv_prey, step=utils.findFairTick(max_x_lv_prey, 10, A_agreeable_tick_size)))
    second_plot.tick_params(axis='x', rotation=x_ticks_rotation)

    second_plot.set_yticks(ticks=np.arange(0, max_y_lv_predator, step=utils.findFairTick(max_y_lv_predator, 10, A_agreeable_tick_size)))


# //////////
#   SLIDER
# //////////

food_axis = plt.axes([0.05, 0.10, 0.85, 0.03], facecolor='lightgreen')
food_slider = Slider(food_axis, 'Food', 0.999, 7.0, valinit=0.0, facecolor='green', dragging=True)


def foodSliderToFood(value):
    food_set = 0
    if value >= 1.0:
        food_set = utils.float2log10int(value)
    return food_set


def setSlider(value):
    food_slider.set_val(value)


def updateSlider(value):
    food_set = foodSliderToFood(value)
    food_slider.valtext.set_text(utils.significantFormat(food_set))
    figure.canvas.draw_idle()


setSlider(food_slider.valinit)
food_slider.on_changed(updateSlider)

# /////////
#   BUTTON
# /////////

next_axis = plt.axes([0.8, 0.025, 0.1, 0.04])
next_button = Button(next_axis, 'Next', color='powderblue', hovercolor='blue')


def next_step(event, **kwargs):
    count = kwargs.get('count', 1)
    for i in range(count):

        #TODO MAKE MORE PRIVATE FIELDS/METHODS IN CLASSES

        #TODO GUI VALUE 1.0-7.0
        #TODO GUI->INTERF 10K
        #TODO GUI->SIM 10K
        #TODO GUI->MODEL 0-1

        food = foodSliderToFood(food_slider.val)# TODO NORMALISATION AS FUNCTION
        sv.step(food, 0)

        #TODO NORMALISATION INSIDE SM.ACT SM.ACT INTO NEW METHOD SM.DECIDE???
        obs = sv.get()
        obs_scaled = [((math.log(o, 10)-1)/6 if o > 0 else 0) for o in obs]
        action_scaled = sm.act(obs_scaled) #TODO NORMALISATION AS FUNCTION
        action_slider = [a*6+1 for a in action_scaled]
        suggestion = math.pow(10, action_slider[0])
        print(suggestion)
        print(obs)
        print(sv.simulation.step_function(obs, [suggestion*1.1]))
        setSlider(action_slider[0])

    plot()
    plt.draw()


next_button.on_clicked(next_step)


def reset(entities):
    global marker_index
    marker_index = 0
    first_plot.cla()
    second_plot.cla()
    sv.reset(entities)
    plot()


# /////////
#   KEYBOARD INPUT
# /////////


def handle_pressed_keyboard(event):
    if event.key == 'n':
        next_step(event, count=1)
    if event.key == 'm':
        next_step(event, count=10)
    if event.key == 'r':
        a = np.random.uniform(2.0, 5.0)
        b = np.random.uniform(1.0, a-0.3)

        reset([10**a, 10**b])
        next_step(event, count=100)


figure.canvas.mpl_connect('key_press_event', handle_pressed_keyboard)


plot()
plt.show()

print("STOP")
