###################################################################################################
#   Name        : package-handler.sh                                                              #
#   Authors     : Jack Harrington && Sagarbir Bandesha                                            #
#   Date        : June 10, 2022                                                                   #
#   Version     : 1.0                                                                             #
#   Description : Gets compatilbe libraries required to run extrapolator.py                       #
###################################################################################################
#!/bin/bash

LIST_OF_LIBS="numpy pandas scikit-learn keras matplotlib"

sudo apt-get update
pip install $LIST_OF_LIBS
