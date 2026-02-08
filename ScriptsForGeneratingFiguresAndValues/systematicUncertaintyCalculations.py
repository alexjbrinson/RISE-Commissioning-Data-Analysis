import numpy as np
import os
import RAP.SpectrumHandler as sh
import RAP.HelperFunctions as hf
amu2eV = np.float64(931494102.42)
electronRestEnergy=510998.950 #in eV


dataDirectory='../Calibration_Data'
directory_contents = os.listdir(dataDirectory)
scans=[int(directory.strip('scan')) for directory in directory_contents if 'scan' in directory]
maxCountsByScan=[sh.importDataFrame(dataDirectory,scan)['PMT0'].max() for scan in scans] #max counts in a single ascii row
maxInstantaneousRate=max(maxCountsByScan)/100/48
print(f'max instantaneous count rate: {maxInstantaneousRate} counts/ns')
print("Detector saturated?", maxInstantaneousRate>0.5) #this count rate should be smaller than 1 count per 2 nanoseconds

#pulling values from scan 16442 - energy corrected
fLines=[]
with open('../results/equal_fwhm_True/cec_sim_toggle_False/mass27/Scan16442/energyCorrected/peakPositions_energyCorrected.txt','r') as file:
  for i, line in enumerate(file):
    if i==0: offset=float(line.lstrip('scan frequency offset: '))
    elif "\tpeak " in line: fLines+=[float(line.split(':')[1].strip("[] "))]
    elif "centroid" in line: centroid=float(line.strip('\tcentroid: '))
fLines = np.array(fLines)

laserFreq=376.052850*3E6; β=(centroid**2-laserFreq**2)/(centroid**2+laserFreq**2)
mass=26.981538408
eLines=hf.freqToVoltage(mass, laserFreq, fLines)
voltage0=29915.8
voltageShifts=eLines-voltage0; #print(voltageShifts) #these are the scan voltages where resonance occured
voltageShifts2=voltageShifts.copy()
voltageShifts2[0]*=(1+1*50E-6) #2->2' 
voltageShifts2[2]*=(1+1*50E-6) #3->2'
voltageShifts2[3]*=(1-1*50E-6) #3->2' #this scaling made negative to maximize centroid deviation
fLines1=hf.voltageToFrequency(mass,laserFreq,voltage0,voltageShifts, colinearity=False)              -offset
fLines2=hf.voltageToFrequency(mass,laserFreq,voltage0,voltageShifts2, colinearity=False)             -offset
fLines3=hf.voltageToFrequency(mass,laserFreq+30,voltage0,voltageShifts, colinearity=False)           -offset
fLines4=hf.voltageToFrequency(mass,laserFreq,voltage0+3.6E-3,voltageShifts, colinearity=False,)      -offset
L=2.1; theta_mis=np.arctan(6E-3/L)+np.arctan(1E-2/L)
fLines5=hf.voltageToFrequency(mass,laserFreq,voltage0,voltageShifts, colinearity=False, theta=0.0076)-offset#, theta=0.0043

def aLowEstimation(fLines):
  return((fLines[0]-fLines[2])/3)

def centroidEstimation(fLines):
  return((5*fLines[0]+7*fLines[3])/12)

def absCentroid(vc, va, β, θc, θa):
  return(np.sqrt(vc*va*(1+β*np.cos(θa))*(1-β*np.cos(θc))))

print("\nSanity Check:")
print('fLines1:',fLines1)#, fLines2, fLines3)
print('aLow estimation1:', aLowEstimation(fLines1)); print('centroid estimation1:', centroidEstimation(fLines1)+offset) #sanity check

print("\nErrors due to scanning voltage:")
print('aLow deviation2:'  , aLowEstimation(fLines1)-aLowEstimation(fLines2), '; simplified:',aLowEstimation(fLines1)*50E-6)
print('centroid deviation:2', centroidEstimation(fLines1)-centroidEstimation(fLines2), '; simplified:', (offset-centroid)*50E-6)

print("\nErrors due to wavemeter offset:")
print('aLow deviation3:'  , aLowEstimation(fLines1)    -    aLowEstimation(fLines3))
print('centroid deviation:3', centroidEstimation(fLines1)-centroidEstimation(fLines3))

print("\nErrors due to beam energy drift during anti/colinear switch:")
print('aLow deviation4:'  , aLowEstimation(fLines1)    -    aLowEstimation(fLines4))
print('centroid deviation:4', centroidEstimation(fLines1)-centroidEstimation(fLines4))

print("\nErrors due to beam misalignment:")
print('aLow deviation5:'  , aLowEstimation(fLines1)    -    aLowEstimation(fLines5))
#this calculation differs from wm, because diff sign on βcos(θ) for anti/colinear
#mean values for non-beam energy corrected centroids from anti/collinear dataset:
vc = 1129900203.5662107
va = 1129899471.7271569
print("centrid deviation5:", absCentroid(vc, va, β, theta_mis, 0) - absCentroid(vc, va, β, 0, 0))