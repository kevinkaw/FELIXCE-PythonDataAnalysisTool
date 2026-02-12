Windows installation:

By default, Windows blocks scripts from running. You need to grant permission once:
1. Open PowerShell as Administrator.
2. Unblock scripts by running: "Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser"
3. Navigate to your folder and run the script: ".\install_FELIXCE_windows.ps1"


Linux installation:

1. Grant Execution Permissions: Open your terminal in the folder containing the script and run: "chmod +x install_FELIXCE_linux.sh"
2. Run the Installer: "./install_FELIXCE_linux.sh"
3. Refresh your Shell: Once finished, you must refresh your shell to recognize the conda command: "source ~/.bashrc"