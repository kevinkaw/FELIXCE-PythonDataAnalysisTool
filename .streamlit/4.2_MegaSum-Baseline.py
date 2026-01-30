import streamlit as st
import matplotlib.pyplot as plt
from packages.BaselineCorrection import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import variables
file_directory = st.session_state.get("file_directory", None)
compiled_data = st.session_state.get("compiled_data", None)
unique_wavenumbers = st.session_state.get("unique_wavenumbers", None)
element1 = st.session_state.get("element1", None)
element2 = st.session_state.get("element2", None)
mass_element1 = st.session_state.get("mass_element1", None)
mass_element2 = st.session_state.get("mass_element2", None)
charge_state = st.session_state.get("charge_state", None)
x_mass = st.session_state.get("x_mass", None)
x_mass_perAtom = st.session_state.get("x_mass_perAtom", None)
MegaSum = st.session_state.get("MegaSum", None)


col1,col2,col3 = st.columns([1,0.1,3]) # col2 is just for spacing

with col1:
    # Complex parameters
    st.markdown("#### Complex parameters")
    st.session_state["n_element1"] = st.number_input(f"size of {element1}", value = st.session_state.get("n_element1",5))
    st.session_state["n_element2"] = st.number_input(f"size of {element2}", value = st.session_state.get("n_element2",1))

    # Baseline parameters
    st.markdown("#### Baseline parameters")
    st.session_state["baseline_reference"] = float(st.text_input("start of baseline in amu",value =st.session_state.get("baseline_reference", 300.0)))
    st.session_state["baseline_width"] = float(st.text_input("width of baseline in amu", value = st.session_state.get("baseline_width", 5.0)))

with col3:
    # Plot parameters
    st.markdown("#### Plot parameters")
    st.session_state["MegaSum_xmin"] = float(st.text_input("Minimum x-value", value = st.session_state.get("MegaSum_xmin", 0.0)))
    st.session_state["MegaSum_xmax"] = float(st.text_input("Maximum x-value", value = st.session_state.get("MegaSum_xmax", 1300)))
    st.session_state["MegaSum_ymin"] = float(st.text_input("Minimum y-value", value = st.session_state.get("MegaSum_ymin",-0.001)))
    st.session_state["MegaSum_ymax"] = float(st.text_input("Maximum y-value", value = st.session_state.get("MegaSum_ymax",0.1))) 

if st.button("✨ Register parameters and make baseline correction!"):

    # Initialize variables
    element1 = st.session_state.get("element1", None)
    element2 = st.session_state.get("element2", None)
    n_element1 = st.session_state.get("n_element1", None)
    n_element2 = st.session_state.get("n_element2", None)
    charge_state = st.session_state.get("charge_state", None)
    baseline_reference = st.session_state.get("baseline_reference", None)
    baseline_width = st.session_state.get("baseline_width", None)
    MegaSum_xmin = st.session_state.get("MegaSum_xmin",None)
    MegaSum_xmax = st.session_state.get("MegaSum_xmax",None)
    MegaSum_ymin = st.session_state.get("MegaSum_ymin",None)
    MegaSum_ymax = st.session_state.get("MegaSum_ymax",None)
    complex = ""
    mass_complex = 0.0
    mass_range_indices = []
    plot_columnIndex_withoutIR = st.session_state.get("plot_columnIndex_withoutIR", -2)
    plot_columnIndex_withIR = st.session_state.get("plot_columnIndex_withIR", -1)

    # get complex properties
    complex, mass_complex, mass_range_indices = mass_range(n_element1,n_element2, element1, element2, mass_element1, mass_element2, charge_state, x_mass)

    # Perform baseline correction
    baseline_correction = baseline(baseline_reference = baseline_reference, \
                                interval = baseline_width, \
                                wavenumber = "MegaSum", \
                                column_withoutIR = MegaSum.columns[plot_columnIndex_withoutIR], \
                                column_withIR = MegaSum.columns[plot_columnIndex_withIR], \
                                data_withoutIR = MegaSum.iloc[:,plot_columnIndex_withoutIR], \
                                data_withIR = MegaSum.iloc[:,plot_columnIndex_withIR], \
                                mass_range= x_mass)
    baseline_range_indices  = baseline_correction.baseline_range()
    baseline_correction.baseline_mean()
    baseline_corrected_data = baseline_correction.baseline_correction()

    # Save variables into memory
    st.session_state["complex"] = complex
    st.session_state["mass_complex"] = mass_complex
    st.session_state["mass_range_indices"] = mass_range_indices
    st.session_state["baseline_range_indices"] = baseline_range_indices

    # update MegaSum x limits
    MegaSum_xmin = mass_complex - 7
    MegaSum_xmax = mass_complex + 7
    st.session_state["MegaSum_xmin"] = MegaSum_xmin
    st.session_state["MegaSum_xmax"] = MegaSum_xmax

    # update MegaSum
    st.session_state["baseline_corrected_MegaSum"] = baseline_corrected_data

    # Create subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, shared_yaxes=True, vertical_spacing=0.05)

    # --- Top plot: raw data ---
    fig.add_trace(
        go.Scatter(
            x=x_mass[mass_range_indices],
            y=MegaSum.iloc[mass_range_indices, plot_columnIndex_withoutIR],
            mode='lines',
            line=dict(color='blue'),
            name=f"MegaSum pre-{MegaSum.columns[plot_columnIndex_withoutIR]}"
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=x_mass[mass_range_indices],
            y=MegaSum.iloc[mass_range_indices, plot_columnIndex_withIR],
            mode='lines',
            line=dict(color='orange'),
            name=f"MegaSum pre-{MegaSum.columns[plot_columnIndex_withIR]}"
        ),
        row=1, col=1
    )

    # Vertical line for mass_complex
    fig.add_trace(
        go.Scatter(
            x=[mass_complex, mass_complex],
            y=[MegaSum_ymin, MegaSum_ymax],
            mode='lines',
            line=dict(color='green', width=1),
            name=complex
        ),
        row=1, col=1
    )

    # Baseline range shading
    fig.add_trace(
        go.Scatter(
            x=x_mass[baseline_range_indices],
            y=[20]*len(baseline_range_indices),
            fill='tozeroy',
            fillcolor='lightgray',
            line=dict(color='lightgray'),
            name='baseline range',
            showlegend=True
        ),
        row=1, col=1
    )

    # Horizontal line at y=0
    fig.add_hline(y=0, line_color='lime', row=1, col=1)

    # --- Bottom plot: baseline corrected ---
    fig.add_trace(
        go.Scatter(
            x=x_mass[mass_range_indices],
            y=baseline_corrected_data.iloc[mass_range_indices,0],
            mode='lines',
            line=dict(color='blue'),
            name='MegaSum post-baseline corrected signal without IR'
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=x_mass[mass_range_indices],
            y=baseline_corrected_data.iloc[mass_range_indices,1],
            mode='lines',
            line=dict(color='orange'),
            name='MegaSum post-baseline corrected signal with IR'
        ),
        row=2, col=1
    )

    # Vertical line for mass_complex
    fig.add_trace(
        go.Scatter(
            x=[mass_complex, mass_complex],
            y=[MegaSum_ymin, MegaSum_ymax],
            mode='lines',
            line=dict(color='green', width=1),
            showlegend=False
        ),
        row=2, col=1
    )

    # Baseline range shading
    fig.add_trace(
        go.Scatter(
            x=x_mass[baseline_range_indices],
            y=[0.1]*len(baseline_range_indices),
            fill='tozeroy',
            fillcolor='lightgray',
            line=dict(color='lightgray'),
            name='baseline range',
            showlegend=False
        ),
        row=2, col=1
    )

    # Horizontal line at y=0
    fig.add_hline(y=0, line_color='lime', row=2, col=1)


    # Top plot: Pre-baseline correction
    fig.add_annotation(
        text="Pre-baseline correction",
        xref="paper", yref="paper",  # use relative coordinates (0 to 1)
        x = mass_complex-4,  # align with the start of the x-axis in the top plot
        y = MegaSum_ymax-2,  # position at the top of the y-axis in the top plot
        showarrow=False,
        font=dict(size=10, color="blue"),
        align="left",
        row=1, col=1
    )

    # Bottom plot: Post-baseline correction
    fig.add_annotation(
        text="Post-baseline correction",
        xref="paper", yref="paper",
        x =mass_complex-4,  # align with the start of the x-axis in the bottom plot
        y = MegaSum_ymax-2,  # position at the top of the y-axis in the bottom plot
        showarrow=False,
        font=dict(size=11, color="green"),
        align="left",
        row=2, col=1
    )


    # --- Update layout ---
    fig.update_layout(
        # Shared x-axis title
        # xaxis=dict(title='Mass (amu)', range=[mass_complex-5, mass_complex+5]),
        xaxis2=dict(title=dict(text='Mass (amu)', font=dict(size=14, color='black')),
                    range=[st.session_state["MegaSum_xmin"], st.session_state["MegaSum_xmax"]],
                    tickfont=dict(size=14, color='black')),  # bottom subplot


        # # Y-axis titles and ranges
        yaxis=dict(title=dict(text='Intensity', font=dict(size=14, color='black')), range=[MegaSum_ymin, MegaSum_ymax], tickfont=dict(size=14, color='black')),  # top subplot
        yaxis2=dict(title=dict(text='Intensity', font=dict(size=14, color='black')), range=[MegaSum_ymin, MegaSum_ymax], tickfont=dict(size=14, color='black')),  # bottom subplot

        showlegend=True,
        legend=dict(font=dict(size=14)),
        height=600
    )

    # Axis limits
    # fig.update_yaxes(range=[MegaSum_ymin, MegaSum_ymax], row=1, col=1)
    # fig.update_yaxes(range=[MegaSum_ymin, MegaSum_ymax], row=2, col=1)
    # fig.update_xaxes(range=[mass_complex-5, mass_complex+5])

    # Display
    st.plotly_chart(fig)

    # Matplotlib
    st.markdown("###### *:green[Static plot with matplotlib]*")
    fig2, ax = plt.subplots(figsize=(10, 4), dpi=600)

    ax.plot(x_mass[mass_range_indices], MegaSum.iloc[mass_range_indices, plot_columnIndex_withoutIR], color='blue', label=f"MegaSum -{MegaSum.columns[plot_columnIndex_withoutIR]}")
    ax.axhline(0, color='lime', linewidth=1)
    st.pyplot(fig2)


