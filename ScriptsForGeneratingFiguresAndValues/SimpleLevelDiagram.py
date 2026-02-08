import matplotlib.pyplot as plt
import os

axesLabelFontSize=18
tickSize=16
labelfontSize=12
fig = plt.figure(figsize=(6,5)); ax=plt.gca()

levelG = 0
levelE = 4.6728909
levelI = 5.985769
levels=[levelG, levelE, levelI]
labels = [r'$\bf{(3s^23p)\,^2P_{1/2}}$', r'$\bf{(3s^25s)\,^2S_{1/2}}$', 'Continuum']

eV2THz=241.798
THz2eV=1/eV2THz
nm2THz=299792.458
THz2nm=1/nm2THz

laser1nm = 265
laser1THz = nm2THz/laser1nm
laser1eV = laser1THz*THz2eV
print(laser1THz)
print(laser1eV)

laser2nm = 532
laser2THz = nm2THz/laser2nm
laser2eV = laser2THz*THz2eV
print(laser2THz)
print(laser2eV)
lasers_eV = [laser1eV,  laser2eV]
lasers_nm = [laser1nm,  laser2nm]
lasers_THz= [laser1THz, laser2THz]
laser_colors = ["Blue", "Green"]

xBounds=[0,1]
yBounds=[0,8]
plt.rcParams["hatch.linewidth"] = 4
continuumRect = plt.Rectangle((xBounds[0],levelI),xBounds[1]-xBounds[0],yBounds[1]-levelI, facecolor="white", edgecolor=(0.5,0.5,0.5,0.5), hatch=r"\\" )
ax.add_patch(continuumRect)

for i,level in enumerate(levels):
  ax.hlines(level, *xBounds, 'k')
  ax.text(sum(xBounds)/2, level+.12, labels[i], fontsize=labelfontSize, horizontalalignment="center", fontweight='bold')
for i, laser_eV in enumerate(lasers_eV):
  # ax.arrow(sum(xBounds)/2-.125+.19*i, levels[i], 0, laser_eV, color=laser_colors[i], length_includes_head=True, head_width=.02, head_length=.25)
  # ax.text(sum(xBounds)/2-.125+.19*i+.01, levels[i]+laser_eV/2-.5, f'{lasers_nm[i]:.0f} nm; {33.356*lasers_THz[i]:.1f} cm'+r'$\bf{^{-1}}$',
  #         color=laser_colors[i], fontsize=labelfontSize, fontweight='bold')
  ax.arrow(.125+.125*i, levels[i], 0, laser_eV, color=laser_colors[i], length_includes_head=True, head_width=.02, head_length=.25)
  ax.text(.125+.125*i+.01, levels[i]+laser_eV/2-.5, f'{lasers_nm[i]:.0f} nm; {lasers_THz[i]:.1f} THz; {33.356*lasers_THz[i]:.1f} cm'+r'$\bf{^{-1}}$',
          color=laser_colors[i], fontsize=labelfontSize, fontweight='bold')
plt.xlim(xBounds)#[0,1.25])
plt.xticks([])
plt.ylim([-0.1,8])
plt.ylabel('Energy (eV)', fontsize=axesLabelFontSize)
plt.yticks(fontsize=tickSize)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
if not os.path.exists('plots'):os.mkdir('plots')
plt.savefig('plots/LevelDiagram.png')
plt.show()