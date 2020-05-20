from typing import Any, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

import utils
import simulation_model
from simulation_view import SimulationView


# ///////////////
#   GUI - INIT
# ///////////////

A_agreeable_tick_size = [1, 2, 4, 5, 10, 20, 25, 40, 50, 75]
sv = SimulationView()

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

    ticks_range = max(max(sv.prey), max(sv.predator))
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

    max_x_lv_prey = max(sv.prey)*1.1
    max_y_lv_predator = max(sv.predator)*1.1
    second_plot.set_xticks(ticks=np.arange(0, max_x_lv_prey, step=utils.findFairTick(max_x_lv_prey, 10, A_agreeable_tick_size)))
    second_plot.tick_params(axis='x',rotation=x_ticks_rotation)

    second_plot.set_yticks(ticks=np.arange(0, max_y_lv_predator, step=utils.findFairTick(max_y_lv_predator, 10, A_agreeable_tick_size)))

# //////////
#   ANFIS
# //////////


model = simulation_model.createModel()

# //////////
#   SLIDER
# //////////

food_min = 0.999
food_max = 7.0
food_axis = plt.axes([0.05, 0.10, 0.85, 0.03], facecolor='lightgreen')
food_slider = Slider(food_axis, 'Food', food_min, food_max, valinit=0.0, facecolor='green', dragging=True)


def foodSliderToFood(value):
    food_set = 0
    if value >= 1.0:
        #food_set = int(value* 10000 / 5)
        food_set = utils.float2log10int(value)
    return food_set


def updateSlider(value):
    food_set = foodSliderToFood(value)
    food_slider.valtext.set_text(utils.significantFormat(food_set))
    figure.canvas.draw_idle()


def updateFoodWithModel():
    food = simulation_model.executeModel(model, sv.simulation.get())
    food = np.clip(food, food_min, food_max)
    food_slider.set_val(food)


updateSlider(food_slider.valinit)
food_slider.on_changed(updateSlider)

# /////////
#   BUTTON
# /////////

next_axis = plt.axes([0.8, 0.025, 0.1, 0.04])
next_button = Button(next_axis, 'Next', color='powderblue', hovercolor='blue')


def next_step(event, **kwargs):
    count = kwargs.get('count', 1)
    update_food = kwargs.get('update_food', False)
    for i in range(count):
        if update_food:
            updateFoodWithModel()
        sv.step(
            foodSliderToFood(food_slider.val),
            0)
    plot()
    plt.draw()


next_button.on_clicked(next_step)

# /////////
#   KEYBOARD INPUT
# /////////


def handle_pressed_keyboard(event):
    if event.key == 'n':
        next_step(event, count=1)
    if event.key == 'm':
        next_step(event, count=10)
    if event.key == 'a':
        next_step(event, count=1, update_food=True)


figure.canvas.mpl_connect('key_press_event', handle_pressed_keyboard)


plot()
plt.show()
