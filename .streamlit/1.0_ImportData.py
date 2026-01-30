import streamlit as st
import h5py
import time
import configparser
import os
from packages.FELIX_HDF5_Reader import *

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
            defaults['file_directory'] = config.get('Import Data', 'file_directory')
            defaults['step_size'] = config.getfloat('Import Data', 'step_size')
        except (configparser.Error, ValueError) as e:
            st.warning(f"Error reading defaults.ini: {e}.")
    return defaults
defaults = load_defaults()

uploaded_files = st.file_uploader("Select HDF5 files to read (adds files to existing list, click `x` to remove)", accept_multiple_files=True, type=["h5"])
st.session_state["file_directory"] = st.text_input("Enter file directory where data is saved. All outputs will be saved here.", value= st.session_state.get("file_directory", defaults['file_directory']))
st.session_state["step_size"] = st.number_input("Enter the step size of your scans.", value=st.session_state.get("step_size", defaults['step_size']))
st.session_state["output_directory"] = os.path.join(st.session_state.get("file_directory"), "output")
# Initialize variables
file_directory = st.session_state.get("file_directory", None)
files = []
raw_data = []
data = []
compiled_data = {}

if uploaded_files:
    for file in uploaded_files:
        files.append(h5py.File(file, "r"))


# Import, read, process H5 files.
if st.button("📖 click this button to import, read, and process the H5 files. Wait for a prompt that files have been loaded to memory."):

    # Validate inputs
    if not uploaded_files:
        st.error("⚠️ Please upload at least one HDF5 file before processing.")
        st.stop()
    
    if not st.session_state.get("step_size"):
        st.error("⚠️ Please enter a valid step size.")
        st.stop()
    
    if st.session_state.get("step_size") <= 0:
        st.error("⚠️ Step size must be greater than 0.")
        st.stop()

    # Show loading animation
    loading_placeholder = st.empty()
    with loading_placeholder:
        st.write("Processing...")
        progress_bar = st.progress(0)

        # Actual file processing with progress updates
        total_steps = len(files) + 4  # Number of steps (reading files + processing steps)
        current_step = 0

    # Read each file (increment progress for each file)
    for file in files:
        # Simulate reading the file
        time.sleep(0.2)  # Add a delay to simulate file loading time if necessary
        current_step += 1
        progress_bar.progress(current_step / total_steps)


    # Actual file processing
    
    # Initialize data
    raw_data = FELIX_HDF5_Reader(files, streamlit_uploaded_files = uploaded_files, step_size=st.session_state.get("step_size"), directory = file_directory) # turn into class object
    current_step += 1
    progress_bar.progress(current_step / total_steps)

    # Import data
    data = raw_data.import_files() # get the wavenumber and signal data
    current_step += 1
    progress_bar.progress(current_step / total_steps)

    # Check data import
    raw_data.check_import() # check output
    st.session_state["dataset_length"] = raw_data.count_rows    
    current_step += 1
    progress_bar.progress(current_step / total_steps)

    # Compile data and visualize wavenumbers
    compiled_data = raw_data.compile_data() # compile data on a per wavenumber basis
    wavenumbers_raw = raw_data.visualize_imported_wavenumbers_raw()
    wavenumbers = raw_data.visualize_imported_wavenumbers()
    unique_wavenumbers = raw_data.visualize_imported_unique_wavenumbers()
    current_step += 1
    progress_bar.progress(current_step / total_steps)

    # Save variables into memory
    st.session_state["raw_data"] = raw_data
    st.session_state["compiled_data"] = compiled_data
    st.session_state["unique_wavenumbers"] = unique_wavenumbers

    # Clear loading animation after processing
    loading_placeholder.empty()
    st.success("Succes! 🤩")

st.markdown("<h3 style='color: red;'>Note: </h3>", unsafe_allow_html=True)
st.markdown("If you navigate between pages, it may seem like your previously imported files are gone. <br><span style='color:blue;'>Rest assured they remain loaded in memory.</span> <br> To clear the memory, click the `kebab menu` icon on the top right and then `clear cache`.", unsafe_allow_html=True)