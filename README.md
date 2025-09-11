# dndc-climate-data-preprocessor-
Streamlit pipeline for converting raw climate data files into DNDC-ready inputs, developed during undergraduate research in atmospheric chemistry at UC Irvine.

# Overview 
This project provides a user-friendly **Streamlit-based data pipeline** for preprocessing meteorological data into DNDC-ready formats.  
It was developed as part of **undergraduate research at UC Irvine**, supporting projects in atmospheric chemistry and sustainability.  

# Features 
 Upload raw meteorological datasets (e.g., NASA POWER CSVs)  
- Map input columns to DNDC-required fields (Jday, MaxT, MinT, Prec, WindSpeed, Humidity)  
- Validate formats, units, and missing values  
- Convert data into DNDC-compatible climate input files  
- Visualize daily and seasonal trends of meteorological variables  
- Download processed DNDC-ready files


# Research Content 
Accurate climate inputs are essential for running the **DNDC (Denitrification-Decomposition) model**, which simulates soil carbon and nitrogen dynamics. This pipeline enables reproducible, standardized preprocessing for research on **atmosphere–soil interactions and air quality impacts**.  

# Tech Stack
- **Languages**: Python  
- **Frameworks**: Streamlit  
- **Libraries**: pandas, numpy 
