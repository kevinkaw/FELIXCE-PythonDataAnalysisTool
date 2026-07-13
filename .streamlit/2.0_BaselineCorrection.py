import streamlit as st
import matplotlib.pyplot as plt
import configparser
import os
from packages.BaselineCorrection import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
            defaults["n_element1"] = config.getint('Complex Parameters', 'n_element1')
            defaults["n_element2"] = config.getint('Complex Parameters', 'n_element2')
            defaults['baseline_reference'] = config.getfloat('Baseline Parameters', 'baseline_reference')
            defaults['baseline_width'] = config.getfloat('Baseline Parameters', 'baseline_width')
            defaults["plot_columnIndex_withoutIR"] = config.getint('Plot Parameters', 'plot_columnIndex_withoutIR')
            defaults["plot_columnIndex_withIR"] = config.getint('Plot Parameters', 'plot_columnIndex_withIR')
            defaults['plot_wavenumber'] = config.getfloat('Plot Parameters', 'plot_wavenumber')
            defaults["baseline_ymax_top"] = config.getfloat('Baseline Parameters', 'baseline_ymax_top')
            defaults["baseline_ymax_bottom"] = config.getfloat('Baseline Parameters', 'baseline_ymax_bottom')
            defaults["mass_xmin"] = config.getfloat('Plot Parameters', 'mass_xmin')
            defaults["mass_xmax"] = config.getfloat('Plot Parameters', 'mass_xmax')

        except (configparser.Error, ValueError) as e:
            st.warning(f"Error reading defaults.ini: {e}.")
    return defaults
defaults = load_defaults()

compiled_data = st.session_state.get("compiled_data", None)
x_mass = st.session_state.get("x_mass", None)
x_mass_perAtom = st.session_state.get("x_mass_perAtom", None)
unique_wavenumbers = st.session_state.get("unique_wavenumbers", None)


col1,col2,col3 = st.columns([0.5,0.5,1]) # col2 is just for spacing

with col1:
    # Complex parameters
    st.markdown("#### Complex parameters")
    st.session_state["n_element1"] = st.number_input("size of main element", value = st.session_state.get("n_element1",defaults["n_element1"]))
    st.session_state["n_element2"] = st.number_input("size of messenger species", value = st.session_state.get("n_element2",defaults["n_element2"]))
with col2:
    # Baseline parameters
    st.markdown("#### Baseline parameters")
    st.session_state["baseline_reference"] = float(st.text_input("start of baseline in amu",value =st.session_state.get("baseline_reference", defaults["baseline_reference"])))
    st.session_state["baseline_width"] = float(st.text_input("width of baseline in amu", value = st.session_state.get("baseline_width", defaults["baseline_width"])))
    st.session_state["baseline_manual_withoutIR"] = (st.text_input("manual baseline value for signal without IR irradiation (optional)", value = st.session_state.get("baseline_manual_withoutIR", None)))
    st.session_state["baseline_manual_withIR"] = (st.text_input("manual baseline value for signal with IR irradiation (optional)", value = st.session_state.get("baseline_manual_withIR", None)))

with col3:
    # Plot parameters
    st.markdown("#### Plot parameters")

    col1, col2 = st.columns([1,1])
    with col1:

        if defaults.get("plot_wavenumber", None) in unique_wavenumbers:
            list_unique_wavenumbers = list(unique_wavenumbers)
            wavenumber_index = list_unique_wavenumbers.index(defaults["plot_wavenumber"])
            st.session_state["plot_wavenumber"] = float(st.selectbox("Select from available wavenumbers", options = unique_wavenumbers, index=wavenumber_index))
        else:
            st.session_state["plot_wavenumber"] = float(st.selectbox("Select from available wavenumbers", options = unique_wavenumbers))
        st.markdown('#')
        st.session_state["mass_xmin"] = float(st.text_input("Mass Spectra: minimum x-value", value = st.session_state.get("mass_xmin", defaults.get("mass_xmin", None))))
        st.session_state["mass_xmax"] = float(st.text_input("Mass Spectra: maximum x-value", value = st.session_state.get("mass_xmax", defaults.get("mass_xmax", None))))
        
    with col2:
        st.session_state["plot_columnIndex_withoutIR"] = int(st.number_input("Column index for signal without IR irradiation", value = st.session_state.get("plot_columnIndex_withoutIR", defaults.get("plot_columnIndex_withoutIR", None))))
        st.session_state["plot_columnIndex_withIR"] = int(st.number_input("Column index for signal with IR irradiation", value = st.session_state.get("plot_columnIndex_withIR", defaults.get("plot_columnIndex_withIR", None))))
        st.session_state["baseline_ymax_top"] = float(st.text_input("Maximum y-value for top plot", value = st.session_state.get("baseline_ymax_top",defaults["baseline_ymax_top"])))
        st.session_state["baseline_ymax_bottom"] = float(st.text_input("Maximum y-value for bottom plot", value = st.session_state.get("baseline_ymax_bottom",defaults["baseline_ymax_bottom"])))

# BUTTON
if st.button("✨ Register parameters and make plot!"):

    # Initialize variables
    element1 = st.session_state.get("element1", None)
    element2 = st.session_state.get("element2", None)
    mass_element1 = st.session_state.get("mass_element1", None)
    mass_element2 = st.session_state.get("mass_element2", None)
    n_element1 = st.session_state.get("n_element1", None)
    n_element2 = st.session_state.get("n_element2", None)
    charge_state = st.session_state.get("charge_state", None)
    plot_columnIndex_withoutIR = st.session_state.get("plot_columnIndex_withoutIR", None)
    plot_columnIndex_withIR = st.session_state.get("plot_columnIndex_withIR", None)
    plot_wavenumber = st.session_state.get("plot_wavenumber",None)
    baseline_reference = st.session_state.get("baseline_reference", None)
    baseline_width = st.session_state.get("baseline_width", None)
    baseline_manual_withoutIR = st.session_state.get("baseline_manual_withoutIR", None)
    baseline_manual_withIR = st.session_state.get("baseline_manual_withIR", None)
    baseline_ymax_top = st.session_state.get("baseline_ymax_top", None)
    baseline_ymax_bottom = st.session_state.get("baseline_ymax_bottom", None)
    
    complex = ""
    mass_complex = 0.0
    mass_range_indices = []

    # get complex properties
    complex, mass_complex, mass_range_indices = mass_range(n_element1,n_element2, element1, element2, mass_element1, mass_element2, charge_state, x_mass)

    # Perform baseline correction
    baseline_correction = baseline(baseline_reference = baseline_reference, \
                                interval = baseline_width, \
                                manual_baseline_withoutIR = baseline_manual_withoutIR, \
                                manual_baseline_withIR = baseline_manual_withIR, \
                                wavenumber = plot_wavenumber, \
                                column_withoutIR = compiled_data[plot_wavenumber].columns[plot_columnIndex_withoutIR], \
                                column_withIR = compiled_data[plot_wavenumber].columns[plot_columnIndex_withIR], \
                                data_withoutIR = compiled_data[plot_wavenumber].iloc[:,plot_columnIndex_withoutIR], \
                                data_withIR = compiled_data[plot_wavenumber].iloc[:,plot_columnIndex_withIR], \
                                mass_range= x_mass)
    baseline_range_indices  = baseline_correction.baseline_range()
    baseline_correction.baseline_mean()
    baseline_corrected_data = baseline_correction.baseline_correction()

    # Save variables into memory
    st.session_state["complex"] = complex
    st.session_state["mass_complex"] = mass_complex
    st.session_state["mass_range_indices"] = mass_range_indices
    st.session_state["baseline_range_indices"] = baseline_range_indices

    tab1, tab2 = st.tabs(["📈 Interactive plot with plotly", "📈 Static plot with matplotlib"])

    with tab1:
        # Create 2-layer Plotly subplot
        fig = make_subplots(rows=2, cols=1, 
                        shared_xaxes=True,
                        vertical_spacing=0.08)

        # Top subplot - Raw data traces
        fig.add_trace(go.Scatter(
            x=x_mass[mass_range_indices], 
            y=compiled_data[plot_wavenumber].iloc[mass_range_indices, plot_columnIndex_withoutIR],
            mode='lines',
            name=compiled_data[plot_wavenumber].columns[plot_columnIndex_withoutIR],
            line=dict(width=3, color='blue'),
            legendgroup="raw"
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=x_mass[mass_range_indices], 
            y=compiled_data[plot_wavenumber].iloc[mass_range_indices, plot_columnIndex_withIR],
            mode='lines',
            name=compiled_data[plot_wavenumber].columns[plot_columnIndex_withIR],
            line=dict(width=3, color="orange"),
            legendgroup="raw"
        ), row=1, col=1)

        # Bottom subplot - Baseline corrected data
        fig.add_trace(go.Scatter(
            x=x_mass[mass_range_indices], 
            y=baseline_corrected_data.iloc[mass_range_indices, 0],
            mode='lines',
            name="baseline corrected signal without IR",
            line=dict(width=3, color='black'),
            legendgroup="corrected"
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=x_mass[mass_range_indices], 
            y=baseline_corrected_data.iloc[mass_range_indices, 1],
            mode='lines',
            name="baseline corrected signal with IR",
            line=dict(width=3, color='red'),
            legendgroup="corrected"
        ), row=2, col=1)

        # Vertical lines for mass complex on both subplots
        fig.add_vline(x=mass_complex, 
                    line_width=2, 
                    line_dash="solid", 
                    line_color="green",
                    annotation_text=complex,
                    annotation_position="top",
                    annotation_font_size=20,
                    annotation_font_color="black",
                    row=1)

        fig.add_vline(x=mass_complex, 
                    line_width=2, 
                    line_dash="solid", 
                    line_color="green",
                    row=2)

        # Baseline range (filled area) for top subplot
        fig.add_trace(go.Scatter(
            x=[x_mass[baseline_range_indices][0], x_mass[baseline_range_indices][-1], 
            x_mass[baseline_range_indices][-1], x_mass[baseline_range_indices][0], 
            x_mass[baseline_range_indices][0]],
            y=[-0.001, -0.001, baseline_ymax_top, baseline_ymax_top, -0.001],
            fill="toself",
            fillcolor='rgba(211,211,211,0.3)',
            line=dict(color='rgba(211,211,211,0.5)', width=1),
            name='baseline range',
            showlegend=True,
            legendgroup="baseline"
        ), row=1, col=1)

        # Baseline range (filled area) for bottom subplot
        fig.add_trace(go.Scatter(
            x=[x_mass[baseline_range_indices][0], x_mass[baseline_range_indices][-1], 
            x_mass[baseline_range_indices][-1], x_mass[baseline_range_indices][0], 
            x_mass[baseline_range_indices][0]],
            y=[-0.001, -0.001, baseline_ymax_bottom, baseline_ymax_bottom, -0.001],
            fill="toself",
            fillcolor='rgba(211,211,211,0.3)',
            line=dict(color='rgba(211,211,211,0.5)', width=1),
            name='baseline range',
            showlegend=False,  # Don't show duplicate legend
            legendgroup="baseline"
        ), row=2, col=1)

        # Horizontal lines at y=0 for both subplots
        fig.add_hline(y=0, 
                    line_width=1, 
                    line_color="lime",
                    row=1)
        
        fig.add_hline(y=0, 
                    line_width=1, 
                    line_color="lime",
                    row=2)

        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            legend=dict(
                x=1.0,                     # Position at far right
                y=0.5,                     # Position at top
                xanchor='right',           # Anchor point for x
                yanchor='top',             # Anchor point for y
                font=dict(size=16),        # Legend font size
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='black',
                borderwidth=1
            )
        )

        # Update x-axes
        fig.update_xaxes(
            range=[mass_complex-5, mass_complex+5],
            title_font=dict(size=18, color='black'),
            tickfont=dict(size=16, color='black'),
            showgrid=False,
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=False,
            row=2, col=1  # Only show x-axis title on bottom plot
        )

        # Update y-axes
        fig.update_yaxes(
            title_text="Intensity (a.u.)",
            title_font=dict(size=18, color='black'),
            tickfont=dict(size=16, color='black'),
            showgrid=False,
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=False
        )

        # Set y-axis ranges for each subplot
        fig.update_yaxes(range=[-0.001, baseline_ymax_top], row=1, col=1)
        fig.update_yaxes(range=[-0.001, baseline_ymax_bottom], row=2, col=1)

        # Add x-axis title only to bottom subplot
        fig.update_xaxes(title_text="Mass (amu)", row=2, col=1)

        # Display the plot
        st.plotly_chart(fig, width="stretch")

    with tab2:
        # Plot raw data and baseline corrected data
        fig,ax = plt.subplots(2,1, dpi=600, sharex=True)

        ax[0].axvline(mass_complex, alpha=0.75,linestyle="solid",linewidth=1, color="green", label = complex)
        ax[0].plot(x_mass[mass_range_indices], compiled_data[plot_wavenumber].iloc[mass_range_indices,plot_columnIndex_withoutIR], label =compiled_data[plot_wavenumber].columns[plot_columnIndex_withoutIR])
        ax[0].plot(x_mass[mass_range_indices], compiled_data[plot_wavenumber].iloc[mass_range_indices,plot_columnIndex_withIR], label = compiled_data[plot_wavenumber].columns[plot_columnIndex_withIR])
        # ax[0].plot(x_mass[:], compiled_data[plot_wavenumber].iloc[:,plot_columnIndex_withoutIR])
        # ax[0].plot(x_mass[:], compiled_data[plot_wavenumber].iloc[:,plot_columnIndex_withIR])
        ax[0].fill_between(x_mass[baseline_range_indices],0.1, color = "lightgray", label = "baseline range")
        ax[0].hlines(0,xmin = x_mass[mass_range_indices][0], xmax =x_mass[mass_range_indices][-1], color="lime")
        ax[0].legend(fontsize = 8)
        
        # Plot baseline corrected data
        ax[1].axvline(mass_complex,alpha=0.75,linestyle="solid",linewidth=1, color="green")
        ax[1].plot(x_mass[mass_range_indices], baseline_corrected_data.iloc[mass_range_indices,0], label = "baseline corrected signal without IR")
        ax[1].plot(x_mass[mass_range_indices], baseline_corrected_data.iloc[mass_range_indices,1], label = "baseline corrected signal with IR")
        ax[1].fill_between(x_mass[baseline_range_indices],0.1, color = "lightgray", label = "baseline range")
        ax[1].hlines(0,xmin = x_mass[mass_range_indices][0], xmax =x_mass[mass_range_indices][-1], color="lime")
        ax[1].legend(fontsize = 8)
        
        # Plot scaling
        ax[0].set_ylim(-0.001, baseline_ymax_top)
        ax[0].set_xlim(mass_complex-5, mass_complex+5)
        ax[1].set_xlim(mass_complex-5, mass_complex+5)
        ax[1].set_ylim(-0.001, baseline_ymax_bottom)

        ax[1].set_xlabel("Mass (amu)")
        fig.supylabel("Intensity (a.u.)")
        # fig.savefig("baseline_correction_plot.png", dpi=600, bbox_inches='tight')
        # print("Saved baseline_correction_plot.png")
        plt.tight_layout()
        
        st.pyplot(fig)
        plt.close(fig)