# RISE Commissioning Repo
This is a snapshot of the analysis code used for the RISE commissioning paper (from Raw MDA files to fit results to paper figures). I only guarantee functionality on windows machines.

Below are the necessary steps to reproduce the analysis yourself:

1. **mda2ascii**
The first thing you need is the mda2ascii program. Downloads for these used to be hosted at ANL.gov, but the webpage seems to have been taken down. MDA_Utilities can still be built locally using the github repo https://github.com/BCDA-APS/MDA_Utilities. I have also included the Windows executable in the "RISEAnalysisPackage" directory. This program should be added to your path.

2. **RISE Analysis Package (RAP)**
The RISE Analysis Package is a (somewhat) abstracted library for analyzing data produced by BECOLA/RISE. The most current version can be found at https://github.com/alexjbrinson/RISEAnalysisPackage, but a snapshot of the package has been embedded in this repo, for posterity. You can install the package with:
    >\>pip install path/to/repo/RISEAnalysisPackage

    Note that the namespace of the package is RAP, e.g. 
    ``` python
     import RAP.HelperFunctions as hf
    ```
3. **Analysis Script**
On the top level of this directory, you should see `ComissioningDataAnalysis.py`. This script will take the raw .MDA files, run the ASCII conversion, construct and export spectra, and fit them. It will also run the anti/collinear analysis to determine $\nu_0$, find the best-fit energy correction calibration function, and then redo the spectrum construction and fitting on the energy-corrected calibration spectra. All of these resulting spectra and fit results will be saved in a subdirectory called "results/".
Note that the initial ascii conversion and dataframe construction is quite slow. For my laptop (48GB RAM, i7-1270P CPU), this takes about 12 minutes to run over all raw data files. The good new is those dataframes only need to be constructed once, so these steps will be skipped over if you ever wish to re-run this analysis script.

4. **Generating Figure Plots**
In the directory "ScriptsForGeneratingFiguresAndValues/", you will find 6 additional scripts, which reproduce the quoted constants, errors, and figures 5-8 from the paper. As Figure 9 requires data from a transition that will be reported on in the future, it is not included here. This data is available upon request.

Note: The scripts in steps 3 and 4 make use of relative paths to import data. They have been written with the expectation that you will run them from the respective directories in which they are saved, e.g.
> path/to/repo>python CommissioningDataAnalysis\.py
> path/to/repo/ScriptsForGeneratingFiguresAndValues>python CalibrationResultsPlots\.py