import streamlit as st
import numpy as np
import configparser
import os

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
            defaults['element1'] = config.get('Complex Parameters', 'element1')
            defaults['element2'] = config.get('Complex Parameters', 'element2')
            defaults['mass_element1'] = config.getfloat('Complex Parameters', 'mass_element1')
            defaults['mass_element2'] = config.getfloat('Complex Parameters', 'mass_element2')
            defaults['charge_state'] = config.get('Complex Parameters', 'charge_state')
            defaults['t_off'] = config.getfloat('Experiment Parameters', 't_off')
            defaults['alpha'] = config.getfloat('Experiment Parameters', 'alpha')
        except (configparser.Error, ValueError) as e:
            st.warning(f"Error reading defaults.ini: {e}.")
    return defaults
defaults = load_defaults()




col1,col2,col3 = st.columns([1,0.1,3]) # col2 is just for spacing

with col1:
    # Species parameters
    st.markdown("### Species parameters")
    st.session_state["element1"] = st.text_input("Element1", value = st.session_state.get("element1", defaults['element1']))
    st.session_state["mass_element1"] = float(st.text_input("Mass of element1 in amu", value = st.session_state.get("mass_element1", defaults['mass_element1'])))
    st.session_state["charge_state"] = st.text_input("Charge state", st.session_state.get("charge_state", defaults['charge_state']))
    st.session_state["element2"] = st.text_input("Element 2", value = st.session_state.get("element2", defaults['element2']))
    st.session_state["mass_element2"] = float(st.text_input("Mass of element2 in amu", value = st.session_state.get("mass_element2", defaults['mass_element2'])))
with col3:
    # Calibration parameters
    st.markdown("### Calibration parameters")
    st.session_state["t_off"] = st.number_input("t_off", value = st.session_state.get("t_off", defaults['t_off']))
    st.session_state["alpha"] = float(st.text_input("alpha", value = st.session_state.get("alpha", defaults['alpha'])))
    st.session_state["dataset_length"] = st.number_input("length of dataset in the time axis", value = st.session_state.get("dataset_length", None))


if st.button("✍️ Register inputs"):

    # Initialize variables
    mass_element1 = st.session_state.get("mass_element1", None)
    t_off = st.session_state.get("t_off", None)
    alpha = st.session_state.get("alpha", None)
    dataset_length = st.session_state.get("dataset_length", None)                             

    # Generate an x-axis
    x_counts=np.linspace(1,dataset_length,dataset_length)
    # Calibrate spectra
    x_mass = alpha*(x_counts - t_off)**2
    x_mass_perAtom = alpha*(x_counts - t_off)**2 / mass_element1


    # Save the variables into memory
    st.session_state["x_mass"] = x_mass
    st.session_state["x_mass_perAtom"] = x_mass_perAtom
    st.success("Inputs registered! 😊")
