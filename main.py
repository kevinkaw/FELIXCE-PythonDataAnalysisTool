import streamlit as st
from st_pages import add_page_title, get_nav_from_toml

st.set_page_config(layout="wide", initial_sidebar_state="expanded")


# Configure dashboard with custom CSS for larger logo
st.markdown("""
<style>
[data-testid="stLogo"] {
    height: 8rem !important;
    margin-bottom: -2rem !important;
    padding-bottom: -2rem !important;
}
[data-testid="stLogo"] img {
    height: 8rem !important;
    width: auto !important;
    margin-bottom: -2rem !important;
    padding-bottom: -2rem !important;
}
</style>
""", unsafe_allow_html=True)

st.logo(r"./logo/logo_FELIXCE_solid.png")
# st.logo("logo_FELIXCE_solid.png", size="large")

# sections = st.sidebar.toggle("Sections", value=True, key="use_sections")

# Load sections
nav = get_nav_from_toml(".streamlit\pages_sections.toml")
pg = st.navigation(nav) # Loads the contents for each entry in the TOML file
add_page_title(pg) # Loads title from each entry in the TOML file
pg.run()