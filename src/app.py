import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

import greendetector

rng = np.random.default_rng()

# Function to generate plot

# Sidebar inputs
minx = st.sidebar.number_input('minx', value=220100.0)
maxx = st.sidebar.number_input('maxx', value=220400.0)
miny = st.sidebar.number_input('miny', value=170600.0)
maxy = st.sidebar.number_input('maxy', value=170900.0)

calculate_button = st.sidebar.button('Calculate')

# Main area
if calculate_button:
    x0 = 720119
    y0 = 6170546
    w, h = 300, 300
    bbox = (x0, y0, x0 + w, y0 + h)
    gplot, gfrac = greendetector.make_green_plot(bbox)
    plot = greendetector.make_tree_plot((minx, miny, maxx, maxy))
    st.pyplot(gplot)
    
    st.sidebar.markdown(f"### KPI")
    st.sidebar.text(f"Vegetation coverage: {(gfrac*100):.1f}%")
    st.sidebar.text(f"Biomass: {(greendetector.BIOMASS_EST0):.1f} m3")
    st.plotly_chart(plot, use_container_width=True)
