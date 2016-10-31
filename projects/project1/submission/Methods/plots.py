# -*- coding: utf-8 -*-
"""a function of ploting figures."""
import numpy as np
import matplotlib.pyplot as plt

# METHOD FOR CREATING A BOXPLOT

def boxplot(x, ymin, ymax, title):
    #TODO : Add Legend for axis

    # Create a figure instance
    fig = plt.figure(1, figsize=(9, 6))

    # Create an axes instance
    ax = fig.add_subplot(111)

    # Create the boxplot
    bp = ax.boxplot(x)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    axes = plt.gca()
    axes.set_ylim([ymin,ymax])
    
    fig.savefig('%s.png'%title)
