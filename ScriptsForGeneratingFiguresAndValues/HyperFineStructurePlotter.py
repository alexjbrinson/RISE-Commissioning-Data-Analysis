import numpy as np
import RAP.HelperFunctions as hf
import matplotlib.pyplot as plt
import pickle, os
from matplotlib import patheffects

axesLabelFontSize=18
tickSize=16
labelfontSize=12
# colorList=["orange","lime","cyan","magenta"]
colorList=['#404040','#d7191c','#ffae61','#1a9641','#0000ff']
linestyleList=['solid','solid', 'solid', 'dashed', 'dotted']

def plotDataAndFit(axis, directory):
  data = np.loadtxt(f'{directory}/spectralData.csv', dtype=float, delimiter=',',skiprows=1)
  xData= data[:,2] ; yData= data[:,3] ; yUncertainty=data[:,4]
  interp_data=np.loadtxt(f'{directory}/_bg.csv', delimiter=','); x_interp=interp_data[:,0]-np.min(interp_data[:,0])+xData[0]; bg_interp=interp_data[:,1]
  iso0_interp=np.loadtxt(f'{directory}/_iso0.csv', delimiter=',')[:,1]; y_interp = bg_interp+iso0_interp
  result=pickle.load(open(f'{directory}/fit_result.pkl','rb'))
  centroid=result['iso0_centroid'].value
  xData -= centroid; x_interp -= centroid

  #plotting
  axis.errorbar(xData,yData,yerr=yUncertainty,fmt='.',ecolor='black',capsize=1,markersize=8, label='data', color='black', alpha=0.5)
  bgLim=230
  axis.plot(x_interp[(x_interp<-615)*(y_interp>bgLim)], y_interp[(x_interp<-615)*(y_interp>bgLim)], color=colorList[1], linestyle=linestyleList[1])
  axis.plot(x_interp[(x_interp<-615)*(y_interp>bgLim)], y_interp[(x_interp<-615)*(y_interp>bgLim)], color=colorList[1], linestyle=linestyleList[1], path_effects=[patheffects.withTickedStroke(spacing=3)], alpha=0.5)
  axis.plot(x_interp[(-615<x_interp)*(x_interp<205)*(y_interp>bgLim)], y_interp[(-615<x_interp)*(x_interp<205)*(y_interp>bgLim)], color=colorList[2], linestyle=linestyleList[2])
  axis.plot(x_interp[(205<x_interp)*(x_interp<880)*(y_interp>bgLim)], y_interp[(205<x_interp)*(x_interp<880)*(y_interp>bgLim)], color=colorList[3], linestyle=linestyleList[3])
  axis.plot(x_interp[(x_interp>880)*(y_interp>bgLim)], y_interp[(x_interp>880)*(y_interp>bgLim)], color=colorList[4], linestyle=linestyleList[4])
  for jj in range(1,len(x_interp)-1):
    if y_interp[jj]<=bgLim: 
      axis.plot(x_interp[jj-1:jj+1], y_interp[jj-1:jj+1], color=colorList[0])

  axis.set_ylabel('Countrate (ion/s)', fontsize=axesLabelFontSize); ax1.set_xlabel('Frequency - Centroid (MHz)', fontsize=axesLabelFontSize)
  axis.legend(loc=1, fontsize=labelfontSize)
  axis.tick_params(axis='both', direction='in')
  plt.xticks(fontsize=tickSize); axis.set_xticks(ticks=[750*i for i in range(-2,3)], labels=[750*i for i in range(-2,3)], fontsize=axesLabelFontSize)
  plt.yticks(fontsize=tickSize); axis.set_yticks(ticks=[500*i for i in range(5)]   , labels=[500*i for i in range(5)]   , fontsize=axesLabelFontSize)
  axis.set_xlim([np.min(data[:,2])-50,np.max(data[:,2])+100])
  axis.set_ylim([np.min(data[:,3])-40,np.max(data[:,3])+100])
  yticks = [*axis.yaxis.get_major_ticks()]
  for i,tick in enumerate(yticks):
      if i==1: tick.set_pad(-40)
      else: tick.set_pad(-50)

def plotHFSDiagram(ax,A1=0,A2=0,iNuc=0,jElec1=0,jElec2=0, centroid=0):
  start=0.00; mid1=0.32; mid2=0.42; end=0.71
  xVals=[start,mid1,mid2,end]
  ax.plot(xVals[0:2],[0,0], color='k'); ax.text(start,0,r'$3P_{1/2}$',color='k', verticalalignment='bottom',horizontalalignment='left',fontsize=16)
  ax.plot(xVals[0:2],[centroid,centroid], color='k'); ax.text(start,centroid,r'$5S_{1/2}$',color='k', verticalalignment='bottom',horizontalalignment='left',fontsize=16)
  f1List=np.arange(abs(iNuc-jElec1),iNuc+jElec1+1,1)
  f2List=np.arange(abs(iNuc-jElec2),iNuc+jElec2+1,1)
  for fTot in f1List:
    shift=hf.energySplitting(A1, 0, iNuc, jElec1, fTot)
    ax.plot(xVals[1:3],[0,shift], color='k'); ax.plot(xVals[2:], [shift,shift], color='k')
    ax.text(.99,shift,f'$F$={int(fTot)}', color='k', verticalalignment='center_baseline',horizontalalignment='right',fontsize=14)
  
  for fTot in f2List:
    shift=hf.energySplitting(A2, 0, iNuc, jElec2, fTot)+centroid
    ax.plot(xVals[1:3],[centroid,shift], color='k'); ax.plot(xVals[2:], [shift, shift], color='k')
    ax.text(.99,shift,f'$F\'$={int(fTot)}', color='k', verticalalignment='center_baseline',horizontalalignment='right',fontsize=14)
  i=0
  for fTot1 in f1List:
    for fTot2 in f2List[::-1]:
      #print(fTot1, fTot2)
      if abs(fTot1-fTot2)<=1:
        #print('viable transition!')
        shift1=hf.energySplitting(A1, 0, iNuc, jElec1, fTot1)
        shift2=hf.energySplitting(A2, 0, iNuc, jElec2, fTot2)+centroid
        xVal=end-((i+.5)/4)*(end-mid2);dx=0;dy=shift2-shift1; #print(dy/10)
        ax.arrow(xVal, shift2-200, dx, 200, color=colorList[-(i+1)], length_includes_head=True, head_width=.03, head_length=200)
        ax.plot([xVal,xVal], [shift1, shift1+dy-200], color=colorList[-(i+1)], linestyle=linestyleList[-(i+1)])
        if i==3:
          ax.plot([xVal,xVal], [shift1, shift1+dy-200], color=colorList[-(i+1)], linestyle=linestyleList[-(i+1)], path_effects=[patheffects.withTickedStroke(spacing=3, angle=135)], alpha=0.5)
        i+=1
  ax.set_xlim([0,1])
  ax.set_yticks([]); ax2.set_xticks([])


if __name__ == '__main__':
  lowestRedChi=np.inf; bestScan=0
  directoryPrefix='../results/equal_fwhm_True/cec_sim_toggle_False/mass27'
  for directory in os.listdir(directoryPrefix):
    if 'Scan' in directory:
      scan=directory.lstrip('Scan')
      redchi=pickle.load(open(f'{directoryPrefix}/{directory}/fit_statistics.pkl','rb'))['redchi']
      if redchi<lowestRedChi: bestScan=scan; lowestRedChi=redchi
  print("scan with lowest red. chi:", bestScan)
  directory=f'{directoryPrefix}/Scan{bestScan}'

  fig1, (ax1,ax2) = plt.subplots(ncols=2, figsize=(8,5), gridspec_kw={'width_ratios': [3, 1],'hspace': 0, 'wspace': 0})
  
  #plotting hfs levels. centroid here is just to control height of diagram. Only the relative spacings between F=2,3 for the ground and excited states are (roughly) to scale 
  plotHFSDiagram(ax2, A1=502.9, A2=135.8,iNuc=5/2,jElec1=1/2,jElec2=1/2, centroid=5000)#centroid here is just to control total height of splitting diagram. Only the 
  #plotting spectrum and best fit
  plotDataAndFit(ax1, directory)

  plt.tight_layout()
  if not os.path.exists('plots'):os.mkdir('plots')
  plt.savefig('plots/exampleSpectrum_Annotated.png')
  plt.show()