Windows installation:

1. Please download and install anaconda or miniconda @ https://repo.anaconda.com/
-- Newbie: install anaconda -- includes many packages that you may or may not use
-- Experienced: install miniconda -- minimal install
---- We are going to setup our own software package environment anyhow so this is the more efficient option.

2. Make sure conda is in your environment PATH variables
-- Press `start` and type `environment variables`
-- At the bottom of the new window, click `Environment Variables`
-- Under `User variables ...` or `System variables ...`, find "Path"
-- Click `edit` and `add` the scripts folder of your anaconda installation.
---- If you didn't change the installation directory, it should be in:
---- C:\Users\<YourProfile>\Anaconda3\Scripts

3. Run "install_FELIXCE_windows.bat"
-- This will check if conda is installed and is in your PATH variables.
-- Then it will make "FELIXCE_v2026.02.12" environment and install packages specified in `environment.yml`

4. Usage
-- For automated launching of FELIXCE, run "run_FELIXCE_windows.bat"
-- For manual launching of FELIXCE:
--- Go to directory with `main.py`
--- activate environment: conda activate FELIXCE_v2026.02.12
--- initialize program: streamlit run main.py

** congratulations! **

5. Other commands
-- To deactivate environment: conda deactivate
-- To uninstall environment: conda remove --name FELIXCE_v2026.02.12 --all
-- list down other variables: conda env list
-- make new conda environment: conda create --name <ENV_NAME>

6. Notes:
-- Since we have all our package dependencies installed in "FELIXCE_v2026.02.12"
---- You must always activate this environment before launching the program (see section4). 
---- Otherwise you will get an error.