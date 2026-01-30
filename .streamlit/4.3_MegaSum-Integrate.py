import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go

# Import variables
file_directory = st.session_state.get("file_directory", None)
compiled_data = st.session_state.get("compiled_data", None)
unique_wavenumbers = st.session_state.get("unique_wavenumbers", None)
element = st.session_state.get("element", None)
mass_element = st.session_state.get("mass_element", None)
charge_state = st.session_state.get("charge_state", None)
messenger = st.session_state.get("messenger", None)
mass_messenger = st.session_state.get("mass_messenger", None)
x_mass = st.session_state.get("x_mass", None)
x_mass_perAtom = st.session_state.get("x_mass_perAtom", None)
baseline_reference = st.session_state.get("baseline_reference", None)
baseline_width = st.session_state.get("baseline_width", None)
complex = st.session_state.get("complex", None)
mass_complex = st.session_state.get("mass_complex", None)
mass_range_indices = st.session_state.get("mass_range_indices", None)
baseline_range_indices = st.session_state.get("baseline_range_indices", None)
MegaSum = st.session_state.get("MegaSum", None)
baseline_corrected_MegaSum = st.session_state.get("baseline_corrected_MegaSum", None)
complex = st.session_state.get("complex", None)
mass_complex = st.session_state.get("mass_complex", None)
mass_range_indices = st.session_state.get("mass_range_indices", None)


col1,col2,col3,col4,col5 = st.columns([1,1,1,1,1])


with col1:
    # Integration parameters
    st.markdown("#### Integration parameters")

    # st.session_state["mass_peaks"] = st.radio("Enter mass peaks, comma separated", options = ["Average mass","Custom input"], index=0)
    options = ["Average mass", "Custom input"]
    selected_option = st.session_state.get("mass_peaks", options[0])  # Default to the first option if not set
    selected_index = options.index(selected_option)  # Get the index of the selected option

    st.session_state["mass_peaks"] = st.radio("Enter mass peaks, comma separated", options=options, index=selected_index)
    mass_peaks = st.session_state.get("mass_peaks", None)

    if mass_peaks == "Average mass":
        
        mass_input = st.text_input("Enter mass peaks, comma separated", value = (mass_complex), label_visibility="collapsed")
        list_mass_isotope = [float(mass_input)]
        st.session_state["list_mass_isotope"] = list_mass_isotope
    else:
        mass_input = st.text_input("Enter mass peaks, comma separated", value = (", ".join(str(x) for x in st.session_state.get("list_mass_isotope", None))), label_visibility="collapsed")
        list_mass_isotope = [float(x.strip()) for x in mass_input.split(",")]
        st.session_state["list_mass_isotope"] = list_mass_isotope
  
    # save input history
    if 'input_history' not in st.session_state:
        st.session_state['input_history'] = []  # List to store input history
    if list_mass_isotope not in st.session_state["input_history"]:
        st.session_state["input_history"].append(list_mass_isotope)
    if len(st.session_state['input_history']) > 5:  # Keep last 5 entries
        st.session_state['input_history'].pop(0)

    with st.popover("Input history"):
        input_history = st.session_state.get("input_history", None)
        for item in input_history:
            st.code(", ".join(str(x) for x in item))


with col2:
    st.markdown("#### ")
    # st.markdown("#### Active mass peak(s):")
    st.write("Active mass peak(s):")
    st.code(", ".join(str(x) for x in list_mass_isotope))
    st.session_state["isotope_scan_width"] = float(st.text_input("Integration width per peak in amu", value = st.session_state.get("isotope_scan_width", 0.3)))
    st.session_state["save_output"] = st.toggle("Save output", value= st.session_state.get("save_output", False))

with col3:
    st.markdown("#### Plot parameters")
    st.session_state["MegaSum_xmin"] = float(st.text_input("Minimum x-value", value = st.session_state.get("MegaSum_xmin", 0.0)))
    st.session_state["MegaSum_xmax"] = float(st.text_input("Maximum x-value", value = st.session_state.get("MegaSum_xmax", 1300)))

with col4:
    st.markdown("####")
    st.session_state["MegaSum_ymin"] = float(st.text_input("Minimum y-value", value = st.session_state.get("MegaSum_ymin",-0.001)))
    st.session_state["MegaSum_ymax"] = float(st.text_input("Maximum y-value", value = st.session_state.get("MegaSum_ymax",0.1))) 
    

with col5:
    st.markdown("####")
    st.session_state["plot_columnIndex_withoutIR"] = int(st.number_input("Column index for MegaSum signal without IR irradiation", value = st.session_state.get("plot_columnIndex_withoutIR", -2)))
    st.session_state["plot_columnIndex_withIR"] = int(st.number_input("Column index for MegaSum signal with IR irradiation", value = st.session_state.get("plot_columnIndex_withIR", -1)))


if st.button("📈Integrate MegaSum!"):

    # Initialize variables
    list_mass_isotope = st.session_state.get("list_mass_isotope")
    isotope_scan_width = st.session_state.get("isotope_scan_width", None)
    x_mass = st.session_state.get("x_mass", None)
    plot_columnIndex_withoutIR = st.session_state.get("plot_columnIndex_withoutIR", None)
    plot_columnIndex_withIR = st.session_state.get("plot_columnIndex_withIR", None)
    x_min2 = st.session_state.get("x_min2", None)
    x_max2 = st.session_state.get("x_max2", None)
    y_min2 = st.session_state.get("y_min2", None)
    y_max2 = st.session_state.get("y_max2", None)
    MegaSum_xmin = st.session_state.get("MegaSum_xmin", None)
    MegaSum_xmax = st.session_state.get("MegaSum_xmax", None)
    MegaSum_ymin = st.session_state.get("MegaSum_ymin", None)
    MegaSum_ymax = st.session_state.get("MegaSum_ymax", None)
    save_output = st.session_state.get("save_output")
    MegaSum = st.session_state.get("MegaSum", None)
    baseline_corrected_MegaSum = st.session_state.get("baseline_corrected_MegaSum", None)

    # Integrate

    #initialize variables
    signal_withIR = []
    signal_withoutIR = []
    newlist_mass_isotope = []
    newlist_scanwidth_isotope = []
    Integral_withoutIR = 0
    Integral_withIR = 0
    list_y1_peaks = []
    list_y2_peaks = []

    # Calculate scan width indices for each isotope
    for isotope in list_mass_isotope:
        scan_width_min = isotope - isotope_scan_width
        scan_width_max = isotope + isotope_scan_width
        scan_width_range_indices = np.where((x_mass >= scan_width_min)&(x_mass <=scan_width_max))[0]
        range_x = x_mass[scan_width_range_indices]
        range_y1 = baseline_corrected_MegaSum.iloc[scan_width_range_indices, plot_columnIndex_withoutIR]
        range_y2 = baseline_corrected_MegaSum.iloc[scan_width_range_indices, plot_columnIndex_withIR]
        peak1 = range_x[np.argmax(range_y1)]
        peak2 = range_x[np.argmax(range_y2)]
        actual_mass_peak = np.mean([peak1,peak2])
        # new_mass = actual_mass_peak # consider both with IR and without IR peaks
        new_mass = peak1
        new_scanwidth = np.where((x_mass >= (new_mass - isotope_scan_width)) & (x_mass <= (new_mass + isotope_scan_width)))[0]
        newlist_mass_isotope.append(new_mass)
        newlist_scanwidth_isotope.append(new_scanwidth)
        signal_withoutIR.append(baseline_corrected_MegaSum.iloc[new_scanwidth, plot_columnIndex_withoutIR])
        signal_withIR.append(baseline_corrected_MegaSum.iloc[new_scanwidth, plot_columnIndex_withIR])
        list_y1_peaks.append(range_y1)
        list_y2_peaks.append(range_y2)
    
    
    # Sum signals over all isotopes
    for index, isotope in enumerate(list_mass_isotope):
        Integral_withoutIR += signal_withoutIR[index].sum()
        Integral_withIR += signal_withIR[index].sum()

    print(list_y1_peaks)
    max_y1value = max([max(peak) for peak in list_y1_peaks])

    st.write(f"### 📊 Integration results for {complex}:")
    st.write(f"- Integrated signal without IR irradiation: **{Integral_withoutIR:.4f}**")
    st.write(f"- Integrated signal with IR irradiation: **{Integral_withIR:.4f}**")

    st.markdown("###### *:green[Interactive plot with plotly]*")
    fig = go.Figure()

    # Main signal plot
    fig.add_trace(go.Scatter(
        x=x_mass,
        y=baseline_corrected_MegaSum.iloc[:, plot_columnIndex_withoutIR],
        mode='lines',
        name='MegaSum without IR'
    ))

    # Average mass vertical line
    fig.add_trace(go.Scatter(
        x=[mass_complex, mass_complex],
        y=[MegaSum_ymin, MegaSum_ymax],
        mode='lines',
        line=dict(color='green', width=2),
        name=f"{complex} average mass"
    ))

    # Multi peak vertical lines and scan width shading
    for index, isotope in enumerate(newlist_mass_isotope):
        fig.add_trace(go.Scatter(
            x=[isotope, isotope],
            y=[MegaSum_ymin, MegaSum_ymax],
            mode='lines',
            line=dict(color='black', width=1),
            showlegend=(index == 0),
            name='Isotope peaks'
        ))

        # Fill the area for scan widths
        fig.add_shape(
            type="rect",
            x0 = isotope - isotope_scan_width,
            x1 = isotope + isotope_scan_width,
            y0 = 0,
            y1 = max_y1value,
            fillcolor='lightgray',
            line=dict(color='rgba(0,0,0,0)'),
            opacity=0.5,
            # showlegend=(index == 0),
            # name='scan width range',
            layer = "below"
        )
    
    # One dummy trace for "scan width range" (appears ONCE) ---
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='lines',
        line=dict(color='lightgray', width=7),
        name='scan width range',
        showlegend=True
    ))


    # Baseline range shading
    fig.add_trace(go.Scatter(
        x=x_mass[baseline_range_indices],
        # y=[0.2]*len(x_mass[baseline_range_indices]),
        y = [max_y1value]*len(x_mass[baseline_range_indices]),
        fill='tozeroy',
        fillcolor='lightsteelblue',
        line=dict(color='lightsteelblue'),
        mode='none',
        name='Baseline range'
    ))

    # Horizontal line for mass range
    fig.add_trace(go.Scatter(
        x=[x_mass[mass_range_indices][0], x_mass[mass_range_indices][-1]],
        y=[0, 0],
        mode='lines',
        line=dict(color='lime', width=2),
        name='zero line'
    ))

    fig.update_layout(
        xaxis=dict(title='Mass (amu)', range=[MegaSum_xmin, MegaSum_xmax]),
        yaxis=dict(title='Intensity', range=[MegaSum_ymin, MegaSum_ymax]),
        legend=dict(font=dict(size=14)),

    )

    st.plotly_chart(fig)


    st.markdown("###### *:green[Static plot with matplotlib]*")
    fig, ax = plt.subplots(figsize=(10, 4), dpi=600)

    ax.axvline(mass_complex,alpha=0.75,linestyle="solid",linewidth=2, color="green", label= f"{complex} average mass")
        
        # multi peak version
    for index, isotope in enumerate(newlist_mass_isotope):
        plt.axvline(isotope,alpha=0.75,linestyle="solid",linewidth=1, color="black")
        plt.fill_between(x_mass[newlist_scanwidth_isotope[index]], max_y1value, color = "lightgray", alpha=0.3, edgecolor=None, zorder=0)


    ax.axvline(0,0, color='black', label='Isotope peaks')
    ax.axvline(0,0, color='lightgray', label='scan width range')
    
    ax.plot(x_mass[:],baseline_corrected_MegaSum.iloc[:,plot_columnIndex_withoutIR], label=f"MegaSum without IR", zorder=5)
    # ax.plot(x_mass[:],MegaSum.iloc[:,plot_columnIndex_withIR], label = f"MegaSum with IR")
    
    ax.fill_between(x_mass[baseline_range_indices],max_y1value, color = "lightsteelblue", label = "baseline range")
    ax.hlines(0,xmin = x_mass[mass_range_indices][0], xmax =x_mass[mass_range_indices][-1], color="lime")

    mass_label = [round(item,2) for item in newlist_mass_isotope]
    textstr = f"Complex: {complex} \nMass peaks: {mass_label} amu \nIntegration width = {isotope_scan_width} amu\nBaseline reference: {baseline_reference} amu \nBaseline width = {baseline_width} amu \nIntegrated signal without IR = {Integral_withoutIR:.4f}"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # Place a text box in the plot
    ax.text(0.01, 0.99, textstr, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    ax.set_xlim(MegaSum_xmin, MegaSum_xmax)
    ax.set_ylim(MegaSum_ymin, MegaSum_ymax)
    ax.set_xlabel("Mass (amu)")
    ax.set_ylabel("Intensity")
    ax.legend(fontsize=10, loc = "upper right")

    fig.tight_layout()
    
    st.pyplot(fig)