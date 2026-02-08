import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import RAP.BeamEnergyAnalysis as bea
import os

axesLabelFontSize=18
tickSize=16
labelfontSize=12

pathPrefix='../results/equal_fwhm_True/cec_sim_toggle_False/CalibrationDiagnostics'
calibFrame_raw=pd.read_csv(f'{pathPrefix}/calibrationFrame_beforeBEC.csv')
calibFrame_bec=pd.read_csv(f'{pathPrefix}/calibrationFrame_afterBEC.csv')
with open(f'{pathPrefix}/calibrationRelevantConstants.txt','r') as file:
  file.readline()
  s=file.readline()
  f0=float(s.lstrip('v0: '))
print(f0)

xVals_raw=np.array(calibFrame_raw['avgScanTime'])
xVals_bec=np.array(calibFrame_bec['avgScanTime'])
print(np.all(xVals_raw==xVals_bec))

xVals=calibFrame_raw['avgScanTime']; xVals-=xVals[0]; xVals/=3600
y1=calibFrame_raw['ΔEkin']; dy1=calibFrame_raw['ΔEkin_uncertainty']
y2=calibFrame_bec['centroid']-f0; dy2=calibFrame_bec['cent_uncertainty'];
y3=calibFrame_bec['aLower']; dy3=calibFrame_bec['aLower_uncertainty']
xRange=np.max(xVals)-np.min(xVals); frac=.05
xMin=np.min(xVals)-frac*xRange
xMax=np.max(xVals)+frac*xRange
x_interp=np.array([xMin,xMax])

cent_ws=np.array(bea.weightedStats(y2,dy2)); sigma_v0 = np.sqrt(np.sum(cent_ws[1:]**2))


centroidAnalysisFrame_corrected=pd.concat([pd.read_csv(f'{pathPrefix}/BEA_ColinearFrame_Corrected.csv'),
                                           pd.read_csv(f'{pathPrefix}/BEA_AnticolinearFrame_Corrected.csv')])
bea_aEstimates=centroidAnalysisFrame_corrected['aLower']; bea_aErrors=centroidAnalysisFrame_corrected['aLower_uncertainty']
bea_aEstimates=np.array(bea_aEstimates); bea_aErrors=np.array(bea_aErrors)

aLow_ws1=np.array(bea.weightedStats(bea_aEstimates,bea_aErrors)); a1=aLow_ws1[0]; sigma_Ag1 = np.sqrt(np.sum(aLow_ws1[1:]**2));

aLow_ws2=np.array(bea.weightedStats(y3,dy3)); a2=aLow_ws2[0]; sigma_Ag2 = np.sqrt(np.sum(aLow_ws2[1:]**2));
print('a1:',a1, sigma_Ag1, aLow_ws1)
print('a2:',a2, sigma_Ag2, aLow_ws2)
rect1 = patches.Rectangle((-len(bea_aEstimates)-.5, a1-sigma_Ag1), len(bea_aEstimates), 2*sigma_Ag1, linewidth=1, facecolor='red', alpha=0.1, edgecolor=(1,0,0,1))
rect2 = patches.Rectangle((-.5, a2-sigma_Ag2), len(y3), 2*sigma_Ag2, linewidth=1, facecolor='blue', alpha=0.1, edgecolor=(0,0,1,1))

plt.figure(figsize=(6,4))
plt.errorbar(x=np.linspace(-len(bea_aEstimates),-1, len(bea_aEstimates)),y=bea_aEstimates,yerr=bea_aErrors, fmt='.',color='r', label='may 2nd')
plt.errorbar(x=np.linspace(0,len(y3)-1, len(y3)),y=y3,yerr=dy3,fmt='.', color='b', label='may 17th')
plt.gca().axhline(y=a1, color='red', linestyle='--')
plt.gca().axhline(y=a2, color='blue', linestyle='--')
plt.gca().add_patch(rect1)
plt.gca().add_patch(rect2)
plt.gca().tick_params(axis='both', direction='in')
plt.gca().set_xticks(ticks=[i for i in range(-len(bea_aEstimates),len(y3))],labels=[], fontsize=tickSize)
plt.xlim([-len(bea_aEstimates)-.5,len(y3)-.5])
plt.xlabel('Scan', fontsize=axesLabelFontSize);
plt.ylabel(r'$A_{^2P_{1/2}}$ (MHz)', fontsize=axesLabelFontSize)
plt.legend(loc=4, fontsize=labelfontSize)

print('v0=',cent_ws[0]+f0,'+/-',sigma_v0, 'std:', np.std(y2))
normalization = (sigma_Ag1**(-2)+sigma_Ag2**(-2))**(-1); sigma_AgMean = normalization**0.5
w1=normalization/sigma_Ag1**2; w2=normalization/sigma_Ag2**2
print("sanity check: w1+w2=", w1+w2)
aBar=w1*a1+w2*a2
print(f"weighted A_lower:{aBar}; naive uncertainty:{sigma_AgMean}")
'''Birge Scaling'''
chiSq = (aBar-a1)**2/sigma_Ag1**2+(aBar-a2)**2/sigma_Ag2**2; birge=np.sqrt(chiSq)
print(f"chiSq = {chiSq}; Birge Factor = {birge}; Birge*sigma_Mean:{birge*sigma_AgMean}")
'''systematic drift parameter'''
sDrift = (((a1-a2)**2-(sigma_Ag1**2+sigma_Ag2**2))/2)**0.5
print(f"Systematic Drift term:{sDrift}")

if not os.path.exists('plots'):os.mkdir('plots')
plt.savefig('plots/SystematicDrift.png')
plt.show()