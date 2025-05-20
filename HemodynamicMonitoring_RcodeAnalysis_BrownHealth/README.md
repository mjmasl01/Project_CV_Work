# Hemodynamic Monitoring – R Analysis of Arterial Catheter Data

## Overview

This repository contains R code, HTML reports, and anonymized datasets supporting the statistical analysis conducted for the peer-reviewed research article:

> Bui D, Hayward G, Chen TH, Apruzzese P, Asher S, Maslow M, Gorgone M, Hunter C, Flaherty D, Kendall M, Maslow A.  
> *Hemodynamic Monitoring In The Cardiac Surgical Patient: Comparison of Three Arterial Catheters.*  
> Journal of Cardiothoracic and Vascular Anesthesia. 2024 May;38(5):1115–1126.  
> DOI: [10.1053/j.jvca.2024.02.010](https://doi.org/10.1053/j.jvca.2024.02.010)  
> PMID: [38461034](https://pubmed.ncbi.nlm.nih.gov/38461034)

---

## My Role

During my time as a Research Assistant with Brown University Health (Summer 2023), I:

- Conducted data cleaning and transformation in RStudio  
- Performed statistical analysis on systolic pressure variations across three arterial catheter sites  
- Used descriptive statistics, linear regression, and logistic regression to support model comparisons  
- Built reproducible workflows in R Markdown  
- Generated HTML reports and data summaries for manuscript inclusion

---

## Repository Structure

HemodynamicMonitoring_RcodeAnalysis_BrownHealth/
│
├── data/ # Anonymized datasets
│ ├── RLdata.csv
│ ├── RSdata.csv
│ ├── BLdata.csv
│ ├── CSVdata.csv
│ └── CSVdata.xlsx
│
├── scripts/ # All R Markdown (.Rmd) analysis scripts
│ ├── Catheter_Use_Comparison.Rmd
│ ├── CopyOfLinearReg---MAP.Rmd
│ ├── CopyOfLogisticReg - MAP.Rmd
│ ├── CopyOfLinearReg - MAP.Rmd
│ ├── LinearReg - Descriptive Statistics - Systolic.Rmd
│ ├── LogisticReg - Descriptive Statistics - Systolic.Rmd
│ ├── HypothesisTest.Rmd
│ ├── Radial Short Data.Rmd
│ ├── Radial Long Data.Rmd
│ └── Brachial Long Data.Rmd
│
├── outputs/ # Rendered HTML reports from analysis
│ ├── Catheter_Use_Comparison.html
│ ├── CopyOfLinearReg---MAP.html
│ ├── CopyOfLogisticReg---MAP.html
│ ├── LinearReg---Descriptive-St...html
│ ├── LogisticReg---Descriptive-S...html
│ ├── Descriptive-Statistics---3-Group-Comparison.html
│ ├── Radial-Short-Data.html
│ ├── Radial-Long-Data.html
│ └── Brachial-Long-Data.html
│
├── Extra Lifespan files/ # Supporting data or alternative files not part of final analysis
│
├── HypothesisTest.pdf # Standalone statistical report summary
├── LifeSpan Data Work.Rproj # RStudio project file
└── README.md # You're reading it

yaml
Copy
Edit

---

## Methods Used

- **Descriptive Statistics**  
- **Hypothesis Testing** (t-tests, ANOVA)  
- **Linear Regression** (`lm`)  
- **Logistic Regression** (`glm`)  
- **Mean Arterial Pressure (MAP) analysis**  
- **Data Visualization** using `ggplot2`

---

## Tools & Technologies

- R, RStudio  
- tidyverse (`dplyr`, `readr`, `ggplot2`, `tidyr`)  
- R Markdown and knitr for reproducible reporting

---

## Citation

If referencing this repository or the associated work, please cite:

> Bui D, Hayward G, Chen TH, Apruzzese P, Asher S, Maslow M, Gorgone M, Hunter C, Flaherty D, Kendall M, Maslow A.  
> *Hemodynamic Monitoring In The Cardiac Surgical Patient: Comparison of Three Arterial Catheters.*  
> Journal of Cardiothoracic and Vascular Anesthesia. 2024 May;38(5):1115–1126.  
> DOI: [10.1053/j.jvca.2024.02.010](https://doi.org/10.1053/j.jvca.2024.02.010)  
> PMID: [38461034](https://pubmed.ncbi.nlm.nih.gov/38461034)

---

## Notes

This repository is for educational and professional demonstration purposes. For questions about the study or access to the full article, please refer to the citation or contact the lead authors listed in the publication.

