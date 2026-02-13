import streamlit as st
from st_pages import add_page_title, get_nav_from_toml

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

# 1. CSS to swap the order of elements in the sidebar
st.markdown("""
<style>
    /* Turn the sidebar content into a flexible column */
    [data-testid="stSidebarContent"] {
        display: flex;
        flex-direction: column;
        padding-top: 1rem !important; /* Adjust if too close to top edge */
    }

    /* Force the Navigation to be 2nd (visually below) */
    [data-testid="stSidebarNav"] {
        order: 2; 
        padding-top: -5rem !important;
        margin-top: -5rem !important;
    }

    /* Force the Logo/Image (and other content) to be 1st (visually above) */
    [data-testid="stSidebarContent"] > div:not([data-testid="stSidebarNav"]) {
        order: 1;
    }
</style>
""", unsafe_allow_html=True)

# 2. Render the logo using sidebar.image (NOT st.logo)
st.sidebar.image(r"./logo/logo_FELIXCE_solid.png", width="stretch")


# Load sections
nav = get_nav_from_toml(r".streamlit/pages_sections.toml")
pg = st.navigation(nav) # Loads the contents for each entry in the TOML file
add_page_title(pg) # Loads title from each entry in the TOML file
pg.run()