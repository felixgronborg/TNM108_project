@ECHO OFF 
:: This batch file installs Jupyter for Python.
TITLE Jupyter Install File
ECHO Please wait... Checking system information.
:: Section 1: :: Install Jupyter Lab Using Pip
ECHO ============================
ECHO INSTALL JUPYTER LAB
ECHO ============================
pip install jupyterlab
:: Section 2: :: Install Jupyter Using Pip.
ECHO ============================
ECHO INSTALL JUPYTER
ECHO ============================
python3 -m pip install jupyter
pip install jupyter
:: Section 3: :: Open Jupyter Notebook
ECHO ============================
ECHO RUN JUPYTER NOTEBOOK
ECHO ============================
jupyter notebook
PAUSE