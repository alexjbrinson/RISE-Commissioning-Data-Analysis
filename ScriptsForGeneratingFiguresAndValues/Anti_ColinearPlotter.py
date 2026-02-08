import numpy as np
import matplotlib.pyplot as plt
import os, pickle

axesLabelFontSize=18
tickSize=16
labelfontSize=12

def loadInterpolation(directory):
  interp_data=np.loadtxt(f'{directory}/_bg.csv', delimiter=','); x_interp=interp_data[:,0]
  bg_interp=interp_data[:,1]; iso0_interp=np.loadtxt(f'{directory}/_iso0.csv', delimiter=',')[:,1]
  y_interp = bg_interp+iso0_interp
  return(x_interp, y_interp)

massNumber=27
mass=26.981538408
laserFreq=3E6*376.052850
iNucList=[2.5]
jGround=0.5; jExcited=0.5
equal_fwhm=True
peakModel='pseudoVoigt'

with open('../results/equal_fwhm_True/cec_sim_toggle_False/CalibrationDiagnostics/calibrationRelevantConstants.txt', 'r') as file:
  for line in file:
    if "v0" in line: finalCentroid= float(line.strip("v0: "))

'''anti/colinear plot'''
fig, (ax1,ax2) = plt.subplots(2, figsize=(6,5),
 gridspec_kw={'height_ratios':[2,1]}, sharex=True)

equal_fwhm=True; cec_sim_toggle=False
colinearRuns     = [16258,16259,16260,16268,16269,16270]
anticolinearRuns = [16253,16254,16255,16263,16264,16265]
colinearCentroids=[]; anticolinearCentroids=[];
colinearMaxAmp=[]; anticolinearMaxAmp=[];
directoryPrefix=f'../results/equal_fwhm_{equal_fwhm}/cec_sim_toggle_{cec_sim_toggle!=False}/mass27/beamEnergy_analysis/'
for i,run in enumerate(np.sort(colinearRuns+anticolinearRuns)):
  colinearity=run in colinearRuns
  directory=f'{directoryPrefix}{"Colinear" if colinearity else "Anticolinear"}/'

  data= np.loadtxt(f'{directory}Scan{run}/spectralData.csv', dtype=float, delimiter=',',skiprows=1)
  data_corrected=np.loadtxt(f'{directory}Scan{run}/energyCorrected/spectralData_energyCorrected.csv', dtype=float, delimiter=',',skiprows=1)
  result=pickle.load(open(f'{directory}Scan{run}/fit_result.pkl','rb'))
  result_corrected=pickle.load(open(f'{directory}Scan{run}/energyCorrected/fit_result_energyCorrected.pkl','rb'))

  cent                 = result['iso0_centroid'].value
  cent_corrected       = result_corrected['iso0_centroid'].value
  cent_uncert          = result['iso0_centroid'].stderr
  cent_corrected_uncert= result_corrected['iso0_centroid'].stderr
  print('cent_corrected:',cent_corrected)

  if colinearity:
    colinearCentroids+=[[cent, cent_uncert]]
    colinearMaxAmp +=[np.max(data[:,2])]
    ax2.errorbar(cent-finalCentroid, y=-i, xerr=cent_uncert, fmt='r.')
    ax2.errorbar(cent_corrected-finalCentroid, y=-i, xerr=cent_corrected_uncert, fmt='r.',alpha=0.5, markerfacecolor='White')

  else:
    anticolinearCentroids += [[cent, cent_uncert]]
    anticolinearMaxAmp +=[np.max(data[:,2])]
    ax2.errorbar(cent-finalCentroid, y=-i, xerr=cent_uncert, fmt='b.')
    ax2.errorbar(cent_corrected-finalCentroid, y=-i, xerr=cent_corrected_uncert, fmt='b.',alpha=0.5, markerfacecolor='White')
colinearCentroids=np.array(colinearCentroids)
anticolinearCentroids=np.array(anticolinearCentroids)

bestScan_Co=colinearRuns[np.argmax(colinearMaxAmp)]
bestScan_Anti=anticolinearRuns[np.argmax(anticolinearMaxAmp)]; print(bestScan_Anti)
directory_co=f'{directoryPrefix}Colinear/Scan{bestScan_Co}'
data_co= np.loadtxt(f'{directory_co}/spectralData.csv', dtype=float, delimiter=',',skiprows=1)
ax1.errorbar(data_co[:,2]-finalCentroid, y=data_co[:,3], yerr=data_co[:,4], fmt='r.')
x_interp_co, y_interp_co = loadInterpolation(directory_co)
x_interp_co=x_interp_co-np.min(x_interp_co[0])+np.min(data_co[:,2])-finalCentroid
ax1.plot(x_interp_co, y_interp_co, color='r', label='colinear')

directory_anti=f'{directoryPrefix}Anticolinear/Scan{bestScan_Anti}'
data_anti= np.loadtxt(f'{directoryPrefix}Anticolinear/Scan{bestScan_Anti}/spectralData.csv', dtype=float, delimiter=',',skiprows=1)
ax1.errorbar(data_anti[:,2]-finalCentroid, y=-data_anti[:,3]+0*np.max(data_co[:,3]), yerr=data_anti[:,4], fmt='b.')

x_interp_anti, y_interp_anti = loadInterpolation(directory_anti)
x_interp_anti=x_interp_anti-np.min(x_interp_anti[0])+np.min(data_anti[:,2])-finalCentroid
ax1.plot(x_interp_anti, -y_interp_anti, color='b', label='anticolinear')

ax1.axvline(x=colinearCentroids[np.argmax(colinearMaxAmp)][0]-finalCentroid,   ymin=0.60, linestyle='--', color='r',alpha=0.5)
ax1.axvline(x=anticolinearCentroids[np.argmax(anticolinearMaxAmp)][0]-finalCentroid, ymax=0.48, linestyle='--', color='b',alpha=0.5)

ax1.tick_params(axis='both', direction='in')
ax2.tick_params(axis='both', direction='in')
ax2.set_ylabel('scan', fontsize=axesLabelFontSize)
ax1.set_yticks(ticks=[400*i for i in range(-2,3)],labels=[abs(400*i) for i in range(-2,3)],fontsize=tickSize)

yticks = [*ax1.yaxis.get_major_ticks()]
for i,tick in enumerate(yticks):
    if i==(len(yticks)-1)/2.0: tick.set_pad(-15)
    else: tick.set_pad(-35)

for i, tick in enumerate(ax1.get_yticklabels()):
  if i>(len(yticks)-1)/2.0: tick.set_color('red')
  elif i< (len(yticks)-1)/2.0: tick.set_color('blue')
ax2.set_yticks(ticks=[-i for i in range(len(colinearRuns+anticolinearRuns))],labels=[], fontsize=tickSize)
ax2.set_xticks(ticks=[1000*i+finalCentroid-finalCentroid for i in range(-2,3)], labels=[1000*i for i in range(-2,3)], fontsize=axesLabelFontSize)
ax1.set_ylabel('Countrate (ion/s)', fontsize=axesLabelFontSize); ax2.set_xlabel('Frequency - Centroid (MHz)', fontsize=axesLabelFontSize)

from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w', label='Collinear',
                          markerfacecolor='r', markersize=10),
                   Line2D([0], [0], marker='o', color='w', label='AntiCollinear',
                          markerfacecolor='b', markersize=10),
                   Line2D([0], [0], marker='o', color='w', label='Co-, corrected',
                          markerfacecolor='w', markeredgecolor='r', markersize=10, alpha=0.5),
                   Line2D([0], [0], marker='o', color='w', label='Anti-, corrected',
                          markerfacecolor='w', markeredgecolor='b', markersize=10, alpha=0.5)]
ax2.legend(handles=legend_elements,loc=1, fontsize=labelfontSize)
plt.xlim([-2038, 2162])
plt.tight_layout()
fig.subplots_adjust(hspace=0)
if not os.path.exists('plots'):os.mkdir('plots')
plt.savefig('plots/Anti_ColinearPlot.png')
plt.show()