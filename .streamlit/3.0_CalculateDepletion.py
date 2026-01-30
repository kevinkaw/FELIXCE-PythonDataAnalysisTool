import streamlit as st
import numpy as np
from packages.IR_yield_Calculator import *
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import os
import configparser

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
            defaults["depletion_ymin"] = config.getfloat('Plot Parameters', 'depletion_ymin')
            defaults["depletion_ymax"] = config.getfloat('Plot Parameters', 'depletion_ymax')
            defaults["IRyield_ymin"] = config.getfloat('Plot Parameters', 'IRyield_ymin')
            defaults["IRyield_ymax"] = config.getfloat('Plot Parameters', 'IRyield_ymax')
            defaults["scan_width"] = config.getfloat('Integration Parameters', 'scan_width')
            defaults["save_output"] = config.getboolean('Integration Parameters', 'save_output')
        except (configparser.Error, ValueError) as e:
            st.warning(f"Error reading defaults.ini: {e}.")
    return defaults
defaults = load_defaults()
file_directory = st.session_state.get("file_directory", None)
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
plot_columnIndex_withoutIR = st.session_state.get("plot_columnIndex_withoutIR", None)
plot_columnIndex_withIR = st.session_state.get("plot_columnIndex_withIR", None)
compilation_baseline_corrected_data = st.session_state.get("compilation_baseline_corrected_data", None)


col1,col2,col3,col4,col5,col6 = st.columns([1,1,0.1,1,1,1]) # col 3 for spacing

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
        list_mass_isotopes = [float(mass_input)]
        st.session_state["list_mass_isotopes"] = list_mass_isotopes
    else:
        mass_input = st.text_input("Enter mass peaks, comma separated", value = (", ".join(str(x) for x in st.session_state.get("list_mass_isotopes", None))), label_visibility="collapsed")
        list_mass_isotopes = [float(x.strip()) for x in mass_input.split(",")]
        st.session_state["list_mass_isotopes"] = list_mass_isotopes
  
    # save input history
    if 'input_history' not in st.session_state:
        st.session_state['input_history'] = []  # List to store input history
    if list_mass_isotopes not in st.session_state["input_history"]:
        st.session_state["input_history"].append(list_mass_isotopes)
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
    st.code(", ".join(str(x) for x in list_mass_isotopes))
    st.session_state["isotope_scan_width"] = float(st.text_input("Integration width per peak in amu", value = st.session_state.get("isotope_scan_width", defaults.get("scan_width", None))))
    st.session_state["save_output"] = st.toggle("Save output", value= st.session_state.get("save_output", defaults.get("save_output", None)))
    

with col4:
    # Plot parameters
    st.markdown("#### Plot parameters")
    st.session_state["plot_columnIndex_withoutIR"] = int(st.number_input("Column index for signal without IR irradiation", value = st.session_state.get("plot_columnIndex_withoutIR", None)))
    st.session_state["plot_columnIndex_withIR"] = int(st.number_input("Column index for signal with IR irradiation", value = st.session_state.get("plot_columnIndex_withIR", None)))
    st.session_state["plot_wavenumber"] = float(st.text_input("Wavenumber to check plots", value = st.session_state.get("plot_wavenumber", None)))
    

with col5:
    st.markdown("#### ")
    st.session_state["mass_xmin"] = float(st.text_input("Mass Spectra: minimum x-value", value = st.session_state.get("mass_xmin", None)))
    st.session_state["mass_xmax"] = float(st.text_input("Mass Spectra: maximum x-value", value = st.session_state.get("mass_xmax", None)))
    st.session_state["mass_ymax"] = float(st.text_input("Mass Spectra: maximum y-value", value = st.session_state.get("mass_ymax", None)))
    

with col6:
    st.markdown("#### ")
    st.session_state["depletion_ymin"] = float(st.text_input("Depletion: minimum y-value", st.session_state.get("depletion_ymin", defaults.get("depletion_ymin", None))))
    st.session_state["depletion_ymax"] = float(st.text_input("Depletion: maximum y-value", st.session_state.get("depletion_ymax", defaults.get("depletion_ymax", None))))
    st.session_state["ln_depletion_ymin"] = float(st.text_input("-ln(depletion): minimum y-value", st.session_state.get("ln_depletion_ymin", defaults.get("IRyield_ymin", None))))
    st.session_state["ln_depletion_ymax"] = float(st.text_input("-ln(depletion): maximum y-value", st.session_state.get("ln_depletion_ymax", defaults.get("IRyield_ymax", None))))

if st.button("✨ Analyze!"):
    # Error checking
    if compilation_baseline_corrected_data is None:
        st.error("No baseline corrected data available.")
        st.stop()
    if unique_wavenumbers is None or len(unique_wavenumbers) == 0:
        st.error("No wavenumbers available.")
        st.stop()
    if x_mass is None:
        st.error("Mass range data not available.")
        st.stop()

    # Initialize variables
    isotope_scan_width = st.session_state.get("isotope_scan_width", None)
    mass_xmin = st.session_state.get("mass_xmin", None)
    mass_xmax = st.session_state.get("mass_xmax", None)
    mass_ymax = st.session_state.get("mass_ymax", None)
    depletion_ymin = st.session_state.get("depletion_ymin", None)
    depletion_ymax = st.session_state.get("depletion_ymax", None)
    ln_depletion_ymin = st.session_state.get("ln_depletion_ymin", None)
    ln_depletion_ymax = st.session_state.get("ln_depletion_ymax", None)
    save_output = st.session_state.get("save_output")
    output_directory = st.session_state.get("output_directory", None)

    
    # Calculate depletioin
    # initialize class
    fullrange_IR_yield_spectra = IR_yield_Calculator(mass_peaks = list_mass_isotopes, scan_width = isotope_scan_width, mass_range=x_mass)
    
    for wavenumber in unique_wavenumbers:
        # print(wavenumber)
        fullrange_IR_yield_spectra.wavenumber = wavenumber
        fullrange_IR_yield_spectra.column_withoutIR = compilation_baseline_corrected_data[wavenumber].columns[-2]
        fullrange_IR_yield_spectra.column_withIR = compilation_baseline_corrected_data[wavenumber].columns[-1]
        fullrange_IR_yield_spectra.data_withoutIR = compilation_baseline_corrected_data[wavenumber].iloc[:,-2]
        fullrange_IR_yield_spectra.data_withIR = compilation_baseline_corrected_data[wavenumber].iloc[:,-1]

        fullrange_IR_yield_spectra.make_IR_yield_spectra()


    IR_yield_spectra = fullrange_IR_yield_spectra.IR_yield_spectra
    isotope_mass_peaks = fullrange_IR_yield_spectra.isotope_mass_peaks
    isotope_scanwidths = fullrange_IR_yield_spectra.isotope_scanwidths

    # convert to numpy array for easy plotting.
    data = np.array(IR_yield_spectra)

    if save_output:
        output_name = f"IR_yield_spectra_{complex}_{int(float(data[0,0]))}-{int(float(data[-1,0]))}cm⁻¹.csv"
        output_fullpath = os.path.join(output_directory, output_name)
        IR_yield_spectra.to_csv(output_fullpath, index=False)
        st.success(rf"Output saved @ {output_fullpath}")
    else:
        st.info("Note: save output is currently off.")



    tab1, tab2, tab3= st.tabs(["📈 Mass spectra - specified wavenumber", "💥 Depletion - full range", "💥 -ln(depletion) - full range"])

    with tab1:

        st.markdown("###### *:green[Interactive plot with plotly]*")
        # Plot via plotly - interactive
        fig = go.Figure()

        # Plot vertical line at mass_complex
        fig.add_trace(go.Scatter(
            x=[mass_complex, mass_complex],
            y=[-0.001, mass_ymax],
            mode='lines',
            line=dict(color="green", width=2, dash="solid"),
            name=str(complex)+" average mass"
        ))

 # Add isotope peaks and scanwidths
        for index, isotope in enumerate(isotope_mass_peaks):
            # Add vertical line for each isotope
            fig.add_trace(go.Scatter(
                x=[isotope, isotope],
                y=[0, mass_ymax],  # Adjust the y range according to your data
                mode='lines',
                line=dict(color='black', width=2),
                showlegend=False
            ))
            
            # Fill the area for scan widths
            fig.add_shape(
                type="rect",
                x0 = isotope - isotope_scan_width,
                x1 = isotope + isotope_scan_width,
                y0 = 0,
                y1 = mass_ymax,
                fillcolor='lightgray',
                line=dict(color='rgba(0,0,0,0)'),
                opacity=0.4,
                name='Fill',
                layer = "below"
            )

        fig.add_trace(go.Scatter(
            x=[0,0],  # Use None for x to avoid displaying a line
            y=[0,0],  # Use None for y to avoid displaying a line
            mode='lines',  # Mode can be anything; it won't show
            name='Isotope peaks',  # This will appear in the legend
            line=dict(color='black', width=2)  # Invisible line
        ))

        fig.add_trace(go.Scatter(
            x=[0,0],  # Use None for x to avoid displaying a line
            y=[0,0],  # Use None for y to avoid displaying a line
            mode='lines',  # Mode can be anything; it won't show
            name='scan width range',  # This will appear in the legend
            line=dict(color='lightgray', width=3)  # Invisible line
        ))



        # Plot data for "without IR" signal
        fig.add_trace(go.Scatter(
            x=x_mass[:],
            y=compilation_baseline_corrected_data[plot_wavenumber].iloc[:, plot_columnIndex_withoutIR],
            mode='lines',
            name=compilation_baseline_corrected_data[plot_wavenumber].columns[plot_columnIndex_withoutIR],
            line=dict(color= "#1f77b4"),
            opacity=1
        ))
        
        # Plot data for "with IR" signal
        fig.add_trace(go.Scatter(
            x=x_mass[:],
            y=compilation_baseline_corrected_data[plot_wavenumber].iloc[:, plot_columnIndex_withIR],
            mode='lines',
            name=compilation_baseline_corrected_data[plot_wavenumber].columns[plot_columnIndex_withIR],
            line=dict(color="#ff7f0e"),
            opacity=1
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
            title=complex,
            legend=dict(x=0.8, y=0.9)
        )

        # Display plot in Streamlit
        st.plotly_chart(fig)


        # matplotlib static plot
        st.markdown("###### *:green[Static plot with matplotlib]*")
        fig, ax = plt.subplots(figsize=(5, 3),dpi=300)

        ax.axvline(mass_complex,alpha=0.75,linestyle="solid",linewidth=2, color="green", label= f"{complex} average mass")
        
        # multi peak version
        for index, isotope in enumerate(isotope_mass_peaks):
            plt.axvline(isotope,alpha=0.75,linestyle="solid",linewidth=1, color="black")
            plt.fill_between(x_mass[isotope_scanwidths[index]],0.5, color = "lightgray")


        ax.axvline(0,0, color='black', label='Isotope peaks')
        ax.axvline(0,0, color='lightgray', label='scan width range')
        
        ax.plot(x_mass[:],compilation_baseline_corrected_data[plot_wavenumber].iloc[:,plot_columnIndex_withoutIR], label=f"{compilation_baseline_corrected_data[plot_wavenumber].columns[plot_columnIndex_withoutIR]}")
        ax.plot(x_mass[:],compilation_baseline_corrected_data[plot_wavenumber].iloc[:,plot_columnIndex_withIR], label = f"{compilation_baseline_corrected_data[plot_wavenumber].columns[plot_columnIndex_withIR]}")
        
        ax.fill_between(x_mass[baseline_range_indices],0.2, color = "lightsteelblue", label = "baseline range")
        ax.hlines(0,xmin = x_mass[mass_range_indices][0], xmax =x_mass[mass_range_indices][-1], color="lime")

        ax.set_xlim(mass_xmin, mass_xmax)
        ax.set_ylim(-0.001, mass_ymax)
        # ax.set_ylim(-0.01, y_max2)
        ax.set_xlabel("Mass (amu)")
        ax.set_ylabel("Intensity")
        ax.legend(fontsize=6, loc = "upper left")

        fig.tight_layout()
        # fig.savefig("MassSpectra.png", dpi=1000)
        
        st.pyplot(fig)
    

    with tab2:
        # convert to numpy array for easy plotting.
        data = np.array(IR_yield_spectra)

        # Interactive plot - plotly
        st.markdown("###### *:green[Interactive plot with plotly]*")
        
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x = data[:,0],
            y = data[:,3],
            name = IR_yield_spectra.columns[3],
            line=dict(color= "#1f77b4")
        ))

        # Plot horizontal line at y=0
        fig.add_trace(go.Scatter(
            x=[data[0,0],data[-1,0]],
            y=[1, 1],
            mode='lines',
            line=dict(color="lime", width=1),
            name = "zero line"
        ))

        # Update layout for the plot
        fig.update_layout(
            yaxis=dict(range=[depletion_ymin, depletion_ymax]),
            xaxis_title="wavenumber (cm⁻¹)",
            yaxis_title="Intensity",
            title= IR_yield_spectra.columns[3] + " " + complex,
            legend=dict(x=0.8, y=0.9)
        )

        st.plotly_chart(fig)

        # Static plot - matplotlib
        st.markdown("###### *:green[Static plot with matplotlib]*")

        fig, ax = plt.subplots(figsize=(8, 3),dpi=300)

        ax.plot(data[:,0],data[:,3])
        ax.scatter(data[:,0],data[:,3])
        ax.legend([IR_yield_spectra.columns[3]], fontsize=7,  loc = "upper right")
        ax.hlines(1,xmin = data[:,0][0], xmax =data[:,0][-1], color="lime")

        mass_label = [round(item,2) for item in isotope_mass_peaks]
        textstr = f"Complex: {complex} \nMass peaks: {mass_label} amu \nIntegration width = {isotope_scan_width} amu\nBaseline reference: {baseline_reference} amu \nBaseline width = {baseline_width} amu"
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        # Place a text box in the plot
        # ax.text(0.6, 0.25, textstr, transform=plt.gca().transAxes, fontsize=5,
        #         verticalalignment='top', bbox=props)
        ax.text(0.01, 0.25, textstr, transform=plt.gca().transAxes, fontsize=7,
                verticalalignment='top', bbox=props)
        
        ax.set_ylim(depletion_ymin,depletion_ymax)
        ax.set_xlabel("wavenumber (cm⁻¹)")

        st.pyplot(fig)

    with tab3:

        # Interactive plot - plotly
        st.markdown("###### *:green[Interactive plot with plotly]*")
        
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x = data[:,0],
            y = data[:,4],
            name = IR_yield_spectra.columns[4],
            line=dict(color= "#1f77b4")
        ))

        # Plot horizontal line at y=0
        fig.add_trace(go.Scatter(
            x=[data[0,0],data[-1,0]],
            y=[0, 0],
            mode='lines',
            line=dict(color="lime", width=1),
            name = "zero line"
        ))

        # Update layout for the plot
        fig.update_layout(
            yaxis=dict(range=[ln_depletion_ymin, ln_depletion_ymax]),
            xaxis_title="wavenumber (cm⁻¹)",
            yaxis_title="Intensity",
            title= IR_yield_spectra.columns[4] + " " + complex,
            legend=dict(x=0.8, y=0.9)
        )

        st.plotly_chart(fig)

        # Static plot - matplotlib
        st.markdown("###### *:green[Static plot with matplotlib]*")

        fig, ax = plt.subplots(figsize=(8, 3),dpi=300)

        ax.plot(data[:,0],data[:,4])
        ax.scatter(data[:,0],data[:,4])
        ax.legend([IR_yield_spectra.columns[4]], fontsize=7, loc = "upper right")
        ax.hlines(0,xmin = data[:,0][0], xmax =data[:,0][-1], color="lime")

        mass_label = [round(item,2) for item in isotope_mass_peaks]
        textstr = f"Complex: {complex} \nMass peaks: {mass_label} amu \nIntegration width = {isotope_scan_width} amu\nBaseline reference: {baseline_reference} amu \nBaseline width = {baseline_width} amu"
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        # Place a text box in the plot
        ax.text(0.01, 0.99, textstr, transform=plt.gca().transAxes, fontsize=7,
                verticalalignment='top', bbox=props)

        ax.set_ylim(ln_depletion_ymin,ln_depletion_ymax)
        ax.set_xlabel("wavenumber (cm⁻¹)")

        st.pyplot(fig)

    st.markdown(f"#### Full range depletion data {complex}:")
    st.table(IR_yield_spectra)