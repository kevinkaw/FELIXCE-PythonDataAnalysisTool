import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from packages.BaselineCorrection import *
import plotly.graph_objs as go
import os
import configparser

# Import variables
# Import variables
def load_defaults():
    """Load default values from defaults.ini file"""
    config = configparser.ConfigParser()
    defaults_file = r'./.streamlit/defaults.ini'  # or provide full path
    defaults= {}
    if os.path.exists(defaults_file):
        try:
            config.read(defaults_file)
            # Update defaults with values from file
            defaults["plot_wavenumber"] = config.getfloat('Plot Parameters', 'plot_wavenumber')
            defaults["mass_xmin"] = config.getfloat('Plot Parameters', 'mass_xmin')
            defaults["mass_xmax"] = config.getfloat('Plot Parameters', 'mass_xmax')
            defaults["mass_ymax"] = config.getfloat('Plot Parameters', 'mass_ymax')
            defaults["plot_columnIndex_withoutIR"] = config.getint('Plot Parameters', 'plot_columnIndex_withoutIR')
            defaults["plot_columnIndex_withIR"] = config.getint('Plot Parameters', 'plot_columnIndex_withIR')
        except (configparser.Error, ValueError) as e:
            st.warning(f"Error reading defaults.ini: {e}.")
    return defaults
defaults = load_defaults()

file_directory = st.session_state.get("file_directory", None)
compiled_data = st.session_state.get("compiled_data", None)
unique_wavenumbers = st.session_state.get("unique_wavenumbers", None)
x_mass = st.session_state.get("x_mass", None)
x_mass_perAtom = st.session_state.get("x_mass_perAtom", None)
baseline_reference = st.session_state.get("baseline_reference", None)
baseline_width = st.session_state.get("baseline_width", None)
complex = st.session_state.get("complex", None)
mass_complex = st.session_state.get("mass_complex", None)
mass_range_indices = st.session_state.get("mass_range_indices", None)
baseline_range_indices = st.session_state.get("baseline_range_indices",None)
plot_wavenumber = st.session_state.get("plot_wavenumber", None)


if st.button("**:blue[#1]** ✨ Perform baseline correction - full range"):

    # Initialize variable
    compilation_baseline_corrected_data = {}
    baseline_correction_fullrange = {}

    baseline_correction_fullrange = baseline(baseline_reference = baseline_reference, interval = baseline_width, mass_range=x_mass)
    compilation_baseline_corrected_data = baseline_correction_fullrange.baseline_correction_fullrange_sum(compiled_data, unique_wavenumbers, baseline_range_indices)

    # export mass spectra
    export = pd.DataFrame({
        "Mass (amu)": x_mass,
        "Mass per atom (amu)": x_mass_perAtom,
        compilation_baseline_corrected_data[plot_wavenumber].columns[-2]: compilation_baseline_corrected_data[plot_wavenumber].iloc[:, -2],
        compilation_baseline_corrected_data[plot_wavenumber].columns[-1]: compilation_baseline_corrected_data[plot_wavenumber].iloc[:, -1]
    })
    # export.to_csv(rf"{file_directory}/output/MassSpectra_{complex}_{plot_wavenumber}cm-1.csv", index=False)
    st.session_state["export_baseline_corrected_data"] = export

    st.session_state["compilation_baseline_corrected_data"] = compilation_baseline_corrected_data
    st.success("Success! 😎")
    st.write(compilation_baseline_corrected_data[plot_wavenumber].head())
    # In case you run into the known bug, comment out `st.table ....` and uncomment the 3 lines below to see the output in the terminal
    # print("\n")
    # pd.set_option('display.max_columns', None)
    # print(compilation_baseline_corrected_data[plot_wavenumber].head())

if st.button("**:blue[#2]** 🚢 Export baseline corrected data"):
    if "export_baseline_corrected_data" not in st.session_state:
        st.error("⚠️ Please perform baseline correction first.")
        st.stop()
    else:
        export = st.session_state.get("export_baseline_corrected_data", None)
        output_directory = st.session_state.get("output_directory", None)
        output_name = f"MassSpectra_{complex}_{plot_wavenumber}cm⁻¹.csv"
        output_fullpath = os.path.join(output_directory, output_name)
        export.to_csv(output_fullpath, index=False)
        st.success(rf"Exported baseline corrected mass spectra @ {output_fullpath}")

st.markdown("#### Plot parameters")

col1, col2, col3, col4, col5 = st.columns([0.4,0.05,0.7,0.05, 1])
with col1:
    st.session_state["plot_wavenumber"] = float(st.text_input("Wavenumber to check plots", value = st.session_state.get("plot_wavenumber", defaults.get("plot_wavenumber", None))))
    st.session_state["mass_ymax"] = float(st.text_input("Maximum y-value", value = st.session_state.get("mass_ymax", defaults.get("mass_ymax", None))))
with col3:
    st.session_state["mass_xmin"] = float(st.text_input("Minimum x-value", value = st.session_state.get("mass_xmin", defaults.get("mass_xmin", None))))
    st.session_state["mass_xmax"] = float(st.text_input("Maximum x-value", value = st.session_state.get("mass_xmax", defaults.get("mass_xmax", None))))
with col5:
    st.session_state["plot_columnIndex_withoutIR"] = int(st.number_input("Column index for signal without IR irradiation", value = st.session_state.get("plot_columnIndex_withoutIR", defaults.get("plot_columnIndex_withoutIR", None))))
    st.session_state["plot_columnIndex_withIR"] = int(st.number_input("Column index for signal with IR irradiation", value = st.session_state.get("plot_columnIndex_withIR", defaults.get("plot_columnIndex_withIR", None))))

    



if st.button("**:blue[#3]** 🚀 Plot full range data"):

    # Initialize variables
    plot_wavenumber = st.session_state.get("plot_wavenumber", None)
    plot_columnIndex_withoutIR = st.session_state.get("plot_columnIndex_withoutIR", None)
    plot_columnIndex_withIR = st.session_state.get("plot_columnIndex_withIR", None)
    compilation_baseline_corrected_data = st.session_state.get("compilation_baseline_corrected_data", None)
    mass_ymax = st.session_state.get("mass_ymax", None)
    mass_xmin = st.session_state.get("mass_xmin", None)
    mass_xmax = st.session_state.get("mass_xmax", None)

    # Plot via matplotlib - static
    # fig, ax = plt.subplots()
    # plt.ion()
    # ax.axvline(mass_complex,alpha=0.75,linestyle="solid",linewidth=1, color="green", label = complex)
    # ax.plot(x_mass[mass_range_indices],compilation_baseline_corrected_data[check_wavenumber].iloc[mass_range_indices,plot_columnIndex_withoutIR], label = compilation_baseline_corrected_data[check_wavenumber].columns[plot_columnIndex_withoutIR])
    # ax.plot(x_mass[mass_range_indices],compilation_baseline_corrected_data[check_wavenumber].iloc[mass_range_indices,plot_columnIndex_withIR], label = compilation_baseline_corrected_data[check_wavenumber].columns[plot_columnIndex_withIR])
    # ax.fill_between(x_mass[baseline_range_indices],0.2, color = "lightgray", label = "baseline range")
    # ax.hlines(0,xmin = x_mass[mass_range_indices][0], xmax =x_mass[mass_range_indices][-1], color="lime")
    # ax.set_ylim(-0.001, y_max2)
    # ax.legend()
    # st.pyplot(fig)
    # plt.close(fig)


    # Plot via plotly - interactive

    fig = go.Figure()

    # Plot data for "without IR" signal
    fig.add_trace(go.Scatter(
        x=x_mass[:],
        y=compilation_baseline_corrected_data[plot_wavenumber].iloc[:, plot_columnIndex_withoutIR],
        mode='lines',
        name=compilation_baseline_corrected_data[plot_wavenumber].columns[plot_columnIndex_withoutIR],
        line=dict(color="#1f77b4"),
        opacity=0.75
    ))
    
    # Plot data for "with IR" signal
    fig.add_trace(go.Scatter(
        x=x_mass[:],
        y=compilation_baseline_corrected_data[plot_wavenumber].iloc[:, plot_columnIndex_withIR],
        mode='lines',
        name=compilation_baseline_corrected_data[plot_wavenumber].columns[plot_columnIndex_withIR],
        line=dict(color="#ff7f0e"),
        opacity=0.75
    ))

    # Plot vertical line at mass_complex
    fig.add_trace(go.Scatter(
        x=[mass_complex, mass_complex],
        y=[-0.001, mass_ymax],
        mode='lines',
        line=dict(color="green", width=1, dash="solid"),
        name=str(complex)
    ))


    # Add rectangle region for baseline range
    fig.add_shape(
        type="rect",
        x0=min(x_mass[baseline_range_indices]),  # Starting x coordinate
        y0=0,                                    # Starting y coordinate
        x1=max(x_mass[baseline_range_indices]),  # Ending x coordinate
        y1=mass_ymax,                               # Ending y coordinate
        fillcolor="lightsteelblue",                   # Fill color
        line=dict(color='rgba(0,0,0,0)'),            # Border color
        opacity = 0.4,
        layer = "below"
    )
    
    # Add an invisible scatter trace to represent the rectangle in the legend
    fig.add_trace(go.Scatter(
        x=[0,0],  # Use None for x to avoid displaying a line
        y=[0,0],  # Use None for y to avoid displaying a line
        mode='lines',  # Mode can be anything; it won't show
        name='Baseline Range',  # This will appear in the legend
        line=dict(color='lightsteelblue', width=2)  # Invisible line
    ))

    # Plot horizontal line at y=0
    fig.add_trace(go.Scatter(
        x=[x_mass[mass_range_indices][0], x_mass[mass_range_indices][-1]],
        y=[0, 0],
        mode='lines',
        line=dict(color="lime", width=1),
        name='zero Line'
    ))

    # Update layout for the plot
    fig.update_layout(
        yaxis=dict(range=[-0.001, mass_ymax]),
        xaxis_title="Mass (amu)",
        yaxis_title="Intensity",
        xaxis=dict(range=[mass_xmin, mass_xmax]),
        title=f"Mass spectra of {complex} at {plot_wavenumber} cm⁻¹",
        legend=dict(x=0.8, y=0.9)
    )

    # Display plot in Streamlit
    st.plotly_chart(fig)

st.markdown(
    "###### *:red[Known bug:]* If there are multiple instances of the :blue[same wavenumber per file], Streamlit will give an error in trying to display it.<br>"
    "The problem is with Streamlit; it's fine on the terminal and does not affect the concatenated data in any way.<br>"
    "My proposed solution is to pick a different `wavenumber` to check. See line 76 of `./streamlit/2.1_BaselineCorrectionFullRange.py`.",
    unsafe_allow_html=True
)