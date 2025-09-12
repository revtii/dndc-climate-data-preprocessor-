# dndc-climate-data-preprocessor
Streamlit pipeline for converting raw climate data files into DNDC-ready inputs, developed during undergraduate research in atmospheric chemistry at UC Irvine.

# Screenshot
<img width="2556" height="1239" alt="Screenshot 2025-09-12 235354" src="https://github.com/user-attachments/assets/66a3da3f-dfb9-4c8c-ae72-32cb7584e733" />

# Overview 
This project provides a user-friendly **Streamlit-based data pipeline** for preprocessing meteorological data into DNDC-ready formats.  
Developed as part of **undergraduate research at UC Irvine**, supporting projects in atmospheric chemistry and sustainability.  
<img width="2556" height="1239" alt="image" src="https://github.com/user-attachments/assets/5d082e32-d145-40ca-848a-5e9515a02371" />

# Features 
- Upload raw meteorological datasets (e.g., NASA POWER CSVs)  
- Map input columns to DNDC-required fields (Jday, MaxT, MinT, Prec, WindSpeed, Humidity)  
- Validate formats, units, and missing values  
- Convert data into DNDC-compatible climate input files  
- Visualize daily and seasonal trends of meteorological variables  
- Download processed DNDC-ready files

Try uploading a raw climate CSV to see it converted into a DNDC-ready file 
https://dndc-climatedatappr.streamlit.app/

# Research Context 
Accurate climate inputs are essential for running the **DNDC (Denitrification-Decomposition) model**, which simulates soil carbon and nitrogen dynamics. This pipeline enables reproducible, standardized preprocessing for research on **atmosphere–soil interactions and air quality impacts**.  

# Tech Stack
- **Languages**: Python  
- **Frameworks**: Streamlit  
- **Libraries**: pandas, numpy 
