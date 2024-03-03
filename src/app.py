import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Function to generate plot
def generate_plot(minx, maxx, miny, maxy):
    x = np.linspace(minx, maxx, 400)
    y = np.linspace(miny, maxy, 400)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))

    fig, ax = plt.subplots()
    ax.contourf(X, Y, Z)
    return fig

# Sidebar inputs
minx = st.sidebar.number_input('minx', value=-5.0)
maxx = st.sidebar.number_input('maxx', value=5.0)
miny = st.sidebar.number_input('miny', value=-5.0)
maxy = st.sidebar.number_input('maxy', value=5.0)

calculate_button = st.sidebar.button('Calculate')

# Main area
if calculate_button:
    plot = generate_plot(minx, maxx, miny, maxy)
    st.pyplot(plot)
