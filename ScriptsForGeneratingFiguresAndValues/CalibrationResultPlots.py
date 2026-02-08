import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from lmfit import Model
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
aLow_ws=np.array(bea.weightedStats(y3,dy3)); sigma_Ag = np.sqrt(np.sum(aLow_ws[1:]**2));

def linFunc(x,bg=0,slope=0): return(bg+slope*x)

linMod=Model(linFunc)
res=linMod.fit(y1, x=xVals, method='leastsq', weights=1/dy1); print(res.fit_report())
print("slope=",res.params['slope'].value,"+/-",res.params['slope'].stderr)
y_interp=res.eval(x=x_interp)
sig_interp=res.eval_uncertainty(x=x_interp, sigma=1)
fig, (ax1,ax2,ax3) = plt.subplots(3, figsize=(6,6), sharex=True)

ax1.plot(x_interp, y_interp-sig_interp, color='red',alpha=0.1)
ax1.plot(x_interp, y_interp+sig_interp, color='red',alpha=0.1)
ax1.fill_between(x_interp, y_interp+sig_interp, y_interp-sig_interp, color='red',alpha=0.1)
ax1.errorbar(xVals, y=y1, yerr=dy1,fmt='.', color='blue')
ax1.plot(x_interp, y_interp,'--', color='red')
ax1.set_ylabel(r'$\Delta E$ (eV)', fontsize=axesLabelFontSize, va='top')

rect2 = patches.Rectangle((xMin, 0-sigma_v0), xMax-xMin, 2*sigma_v0, linewidth=1, facecolor='red', alpha=0.1, edgecolor=(1,0,0,1))
ax2.add_patch(rect2)
ax2.errorbar(xVals, y=y2, yerr=dy2,fmt='.', color='blue')
ax2.axhline(y=0, color='red', linestyle='--')
ax2.set_ylabel(r'$\nu_0$ (MHz)', fontsize=axesLabelFontSize, va='top')

rect3 = patches.Rectangle((xMin, aLow_ws[0]-sigma_Ag), xMax-xMin, 2*sigma_Ag, linewidth=1, facecolor='red', alpha=0.1, edgecolor=(1,0,0,1))
ax3.add_patch(rect3)
ax3.errorbar(xVals, y=y3, yerr=dy3,fmt='.', color='blue')
ax3.axhline(y=aLow_ws[0], color='red', linestyle='--')
ax3.set_ylabel(r'$A_{^2P_{1/2}}$ (MHz)', fontsize=axesLabelFontSize, va='top')

ax1.tick_params(axis='both', direction='in')
ax2.tick_params(axis='both', direction='in')
ax3.tick_params(axis='both', direction='in')

ax1.set_xlim([xMin, xMax])

ax1.get_yaxis().set_label_coords(-0.13,0.5)
ax2.get_yaxis().set_label_coords(-0.13,0.5)
ax3.get_yaxis().set_label_coords(-0.13,0.5)

plt.xlabel('Time (hours)', fontsize=axesLabelFontSize) 
plt.tight_layout()
if not os.path.exists('plots'):os.mkdir('plots')
plt.savefig('plots/MeasurementStabilityPlot.png')
plt.show();#plt.close()
print(list(calibFrame_bec.columns))