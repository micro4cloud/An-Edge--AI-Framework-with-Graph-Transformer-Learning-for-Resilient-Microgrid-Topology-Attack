import matplotlib.pyplot as plt  
import numpy as np  
 
scenarios = [  

    "10 nodes",  
    "20 nodes",  
    "30 nodes",  
    "40 nodes",  
    "50+ nodes"  
] 
 
ccpa = [88,85,81,77,74]  
pbrd = [90,88,85,82,80]  
bdiv = [91,89,86,84,82]  
egat = [97,97.5,97.8,97.4,97.2]  
 
plt.figure(figsize=(10,5))  
 
plt.plot(scenarios, ccpa, label='CCPA -FLO') 
plt.plot(scenarios, pbrd, label='PBRD')  
plt.plot(scenarios, bdiv, label='BDIV')  
plt.plot(scenarios, egat, label='Proposed EGAT')  
 
plt.ylabel("Detection Accuracy (%)")  
plt.xlabel("Microgrid Scenario")  
 
plt.legend()  
 
plt.grid(True)  
 
plt.show() 
