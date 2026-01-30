import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from packages.BaselineCorrection import *
import plotly.graph_objs as go
import os

# Import variables
file_directory = st.session_state.get("file_directory", None)
output_directory = st.session_state.get("output_directory", None)
compiled_data = st.session_state.get("compiled_data", None)
unique_wavenumbers = st.session_state.get("unique_wavenumbers", None)
element1 = st.session_state.get("element1", None)
element2 = st.session_state.get("element2", None)
mass_element1 = st.session_state.get("mass_element1", None)
mass_element2 = st.session_state.get("mass_element2", None)
charge_state = st.session_state.get("charge_state", None)
x_mass = st.session_state.get("x_mass", None)
x_mass_perAtom = st.session_state.get("x_mass_perAtom", None)
baseline_range_indices = st.session_state.get("baseline_range_indices", None)


st.markdown("##### :red[Caution]: You need to run the program all the way to section 2.0 first.")

if st.button("**:blue[#1]** ➕ Create a sum of all mass spectra + baseline correction! (wavenumber independent)"):
    # Initialize variables
    MegaSum = {}
    MegaTable = pd.DataFrame()
    NewTable = pd.DataFrame()
    baseline_withoutIR = 0
    baseline_withIR = 0

    # assemble everything into a gigantic table
    for wavenumber in unique_wavenumbers:
        MegaTable = pd.concat([MegaTable, compiled_data[wavenumber]],axis=1)

    # sum all the signals without and with IR irradiation
    signal_withoutIR = MegaTable.iloc[: ,0: :2].sum(axis=1)
    signal_withIR = MegaTable.iloc[:, 1: :2].sum(axis=1)

    # baseline correction
    baseline_withoutIR = np.mean(signal_withoutIR[baseline_range_indices])
    baseline_withIR = np.mean(signal_withIR[baseline_range_indices])


    new_table = pd.DataFrame({
        "MegaSum_signal_withoutIR": signal_withoutIR,
        "MegaSum_signal_withIR": signal_withIR,
        "MegaSum_baseline_corrected_signal_withoutIR": signal_withoutIR - baseline_withoutIR,
        "MegaSum_baseline_corrected_signal_withIR": signal_withIR - baseline_withIR
    })

    export_new_table = pd.DataFrame({
        "Mass (amu)": x_mass,
        "MegaSum_signal_withoutIR": signal_withoutIR,
        "MegaSum_signal_withIR": signal_withIR,
        "MegaSum_baseline_corrected_signal_withoutIR": signal_withoutIR - baseline_withoutIR,
        "MegaSum_baseline_corrected_signal_withIR": signal_withIR - baseline_withIR
    })

    # create the mega sum table
    # st.session_state["MegaSum"] = pd.concat([MegaTable, new_table],axis=1)
    # MegaSum = st.session_state.get("MegaSum", None)
    st.session_state["MegaSum"] = new_table
    MegaSum = st.session_state.get("MegaSum", None)
    output_fullpath = os.path.join(output_directory, "MegaSum_table.csv")
    export_new_table.to_csv(output_fullpath, index=False)
    st.success(rf"Created Mega Sum table and exported @ {output_fullpath} ")

    st.write(MegaSum)



st.markdown("#### Plot parameters")
col1,col2,col3,col4 = st.columns([1,1,1,1])

with col1:
    st.session_state["max_element"] = int(st.number_input(f"Maximum size for {element1}", value = st.session_state.get("max_element", 20)))
    st.session_state["max_messenger"] = int(st.number_input(f"Maximum size for {element2}", value = st.session_state.get("max_messenger", 7)))

with col2:
    st.session_state["plot_columnIndex_withoutIR"] = int(st.number_input("Column index for signal without IR irradiation", value = st.session_state.get("plot_columnIndex_withoutIR", -2)))
    st.session_state["plot_columnIndex_withIR"] = int(st.number_input("Column index for signal with IR irradiation", value = st.session_state.get("plot_columnIndex_withIR", -1)))

with col3:
    st.session_state["MegaSum_xmin"] = float(st.text_input("Minimum x-value", value = st.session_state.get("MegaSum_xmin", 0.0)))
    st.session_state["MegaSum_xmax"] = float(st.text_input("Maximum x-value", value = st.session_state.get("MegaSum_xmax", 1300)))

with col4:
    st.session_state["MegaSum_ymin"] = float(st.text_input("Minimum y-value", value = st.session_state.get("MegaSum_ymin",-1)))
    st.session_state["MegaSum_ymax"] = float(st.text_input("Maximum y-value", value = st.session_state.get("MegaSum_ymax",100))) 
    


if st.button("**:blue[#2]** 📈 Plot mega sum!"):

    # initialize variables
    x_mass = st.session_state.get("x_mass", None)
    plot_columnIndex_withoutIR = st.session_state.get("plot_columnIndex_withoutIR", None)
    plot_columnIndex_withIR = st.session_state.get("plot_columnIndex_withIR", None)
    MegaSum_xmin = st.session_state.get("MegaSum_xmin", None)
    MegaSum_xmax = st.session_state.get("MegaSum_xmax", None)
    MegaSum_ymin = st.session_state.get("MegaSum_ymin", None)
    MegaSum_ymax = st.session_state.get("MegaSum_ymax", None)
    MegaSum = st.session_state.get("MegaSum", None)
    element1 = st.session_state.get("element1", None)
    element2 = st.session_state.get("element2", None)
    mass_element1 = st.session_state.get("mass_element1", None)
    mass_element2 = st.session_state.get("mass_element2", None)
    charge_state = st.session_state.get("charge_state", None)
    max_element = st.session_state.get("max_element", None)
    max_messenger = st.session_state.get("max_messenger", None)


    st.markdown("###### *:green[Interactive plot with plotly]*")
    # Create a Plotly figure
    fig = go.Figure()

    # Add horizontal line at y=0
    fig.add_shape(
        type='line',
        x0=x_mass[0], 
        x1=x_mass[-1], 
        y0=0, 
        y1=0,
        line=dict(color="limegreen"),
        layer = "below",
        name = "zero line"
    )

    # Plot the baseline range
    fig.add_trace(go.Scatter(
        x=x_mass[baseline_range_indices],
        y=[(MegaSum_ymax/2)] * len(baseline_range_indices),
        fill='tozeroy',
        mode='lines',
        fillcolor='lightsteelblue',
        line=dict(color='lightsteelblue'),
        name='baseline range',
        opacity = 0.5
    ))

    # Plot the mass spectra for the two specified columns
    fig.add_trace(go.Scatter(
        x=x_mass,
        y=MegaSum.iloc[:, plot_columnIndex_withoutIR],
        mode='lines',
        name=MegaSum.columns[plot_columnIndex_withoutIR]
    ))

    fig.add_trace(go.Scatter(
        x=x_mass,
        y=MegaSum.iloc[:, plot_columnIndex_withIR],
        mode='lines',
        name=MegaSum.columns[plot_columnIndex_withIR]
    ))

    # Main element annotations
    for i in range(1, max_element + 1):
        n = i
        m = 0
        complex, mass_complex, _ = mass_range(n, m, element1, element2, mass_element1, mass_element2, charge_state, x_mass)
        mass_index = np.where((x_mass >= mass_complex - 2) & (x_mass <= mass_complex + 2))[0]
        y1 = np.max(MegaSum.iloc[mass_index, plot_columnIndex_withoutIR])
        y2 = np.max(MegaSum.iloc[mass_index, plot_columnIndex_withIR])
        y = max(y1, y2)

        fig.add_annotation(
            x=mass_complex,
            y=y + 3,
            text=f"({n},{m})",
            showarrow=False,
            arrowhead=2,
            ax=0,
            ay=-(y+100),
            font=dict(size=20, color="red"),
            textangle=90,
        )

    # Complex annotations
    for i in range(1, max_element + 1):
        for j in range(1, max_messenger + 1):
            n = i
            m = j
            complex, mass_complex, _ = mass_range(n, m, element1, element2, mass_element1, mass_element2, charge_state, x_mass)
            mass_index = np.where((x_mass >= mass_complex - 2) & (x_mass <= mass_complex + 2))[0]
            y1 = np.max(MegaSum.iloc[mass_index, plot_columnIndex_withoutIR])
            y2 = np.max(MegaSum.iloc[mass_index, plot_columnIndex_withIR])
            y = max(y1, y2)

            fig.add_annotation(
                x=mass_complex,
                y=y + (0.5*i),
                text=f"({n},{m})",
                showarrow=False,
                arrowhead=2,
                ax=0,
                ay=-(y + (10 * i)),
                font=dict(size=18),
                textangle=90,
            )

    # Set the x and y limits
    fig.update_xaxes(range=[MegaSum_xmin, MegaSum_xmax])
    fig.update_yaxes(range=[MegaSum_ymin, MegaSum_ymax])

    # Set the axis labels
    fig.update_layout(
        xaxis_title="Mass (amu)",
        yaxis_title="Intensity",
        legend=dict(
            font=dict(size=14),  # Font size of the legend
            orientation='v',     # Set legend orientation to horizontal
            yanchor='top',    # Anchor the legend to the bottom
            xanchor='right',    # Center the legend horizontally
        )
    )

    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig)


    st.markdown("###### *:green[Static plot with matplotlib]*")
    fig, ax = plt.subplots(figsize=(7, 4), dpi=600)

    # plot the mass spectra
    ax.axhline(0, color = "limegreen")
    # ax.fill_between(x_mass[baseline_range_indices],0.2, color = "lightsteelblue", label = "baseline range")
    ax.plot(x_mass[:], MegaSum.iloc[:,plot_columnIndex_withoutIR], label = MegaSum.columns[plot_columnIndex_withoutIR])
    # ax.plot(x_mass[:], MegaSum.iloc[:,plot_columnIndex_withIR], label = MegaSum.columns[plot_columnIndex_withIR])

    # combinations of complexes
    
    # main element
    for i in range(1, max_element+1):
        n = i
        m = 0

        complex, mass_complex, _ = mass_range(n,m,element1,element2,mass_element1,mass_element2, charge_state, x_mass)
        mass_index = np.where((x_mass >= mass_complex -2) & (x_mass <= mass_complex+2))[0]
        y1 = np.max(MegaSum.iloc[mass_index, plot_columnIndex_withoutIR])
        y2 = np.max(MegaSum.iloc[mass_index, plot_columnIndex_withIR])
        y = max(y1,y2)
        ax.annotate(f"({n},{m})", (mass_complex, y), textcoords="offset points", xytext=(0,20), ha='center', fontsize=10, rotation = 90, color="red")

    # complex
    for i in range(1,max_element+1):
        for j in range(1, max_messenger+1):
            n=i
            m = j
            complex, mass_complex, _ = mass_range(n,m,element1,element2,mass_element1,mass_element2, charge_state, x_mass)
            mass_index = np.where((x_mass >= mass_complex -2) & (x_mass <= mass_complex+2))[0]
            y1 = np.max(MegaSum.iloc[mass_index, plot_columnIndex_withoutIR])
            y2 = np.max(MegaSum.iloc[mass_index, plot_columnIndex_withIR])
            y = max(y1,y2)
            # ax.annotate(f"({n},{m})", (mass_complex, y), textcoords="offset points", xytext=(0,y+(4*i)), ha='center', fontsize=8, rotation = 90)
            ax.annotate(f"({n},{m})", (mass_complex, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, rotation = 90)


    ax.set_xlim(MegaSum_xmin, MegaSum_xmax)
    ax.set_ylim(MegaSum_ymin, MegaSum_ymax)
    ax.set_xlabel("Mass (amu)")
    ax.set_ylabel("Intensity")
    # ax.legend(fontsize=5)
    fig.tight_layout()
    # fig.savefig("MegaSum_plot.png", dpi=600)

    st.pyplot(fig)
