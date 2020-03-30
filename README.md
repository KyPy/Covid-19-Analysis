# Covid-19-Analysis
Analysis of Covid-19 datasets

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

Assuming the base installation of Ananconda, two additional packages are neccessary: seaborn and bokeh
The code should run with no issues using Python versions 3.*.

## Project Motivation<a name="motivation"></a>

The goal is to better understand influences and durations of the Covid-19 pandemic. The analysis should answer the following questions:

1. How long did it take in China and South Korea to reach the turning point of declining new infections or deaths?
2. Are effects of national restrictions visible in the time series?
3. Is there a correlation of national key figures (e.g. health care capacity) and mortality ratio?

The contained jupyter notebook is linking to the github repository of John Hopkins University, which is updated daily. They can be therefore used to quickly view changes in the pandemic spreading.

## File Descriptions <a name="files"></a>

- analysis.ipynb contains the script, diagrams and the result interpretation
- test.py was used in the development process for testing

The folder data contains the Kaggle dataset about the country information (covid19countryinfo.csv) and the self-researched file of national restrictions (restrictions.csv).

## Results<a name="results"></a>

The main findings of the code can be found at the post available [here](https://medium.com/@kai.sandmann/analysing-influences-on-covid-19-spread-in-major-countries-f74cd7a0f309).

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Big credits to John Hopkins University for making the Covid-19 dataset publicly available on: https://github.com/CSSEGISandData/COVID-19
Also credits to Kaggle user koryto for posting the summary dataset: https://www.kaggle.com/koryto/countryinfo
