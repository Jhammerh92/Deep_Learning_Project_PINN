import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from DiseaseModel import SIRD, SIRDn, SIRDIm, SIRD2Var

data = np.load(r"/Users/jhh/Library/Mobile Documents/com~apple~CloudDocs/DTU/11Semester Msc/Deep Learning/Deep_Learning_Project_PINN/exercises_Jakob/SIRD_sim_data_S100k_I10_a0.2_b0.1.npy")
sird_model =  SIRDn()


sird_model.plot_with_sliders(data=data*10)


plt.show()