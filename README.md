# Sci_Prog_Group_Assignment
# Data Setup:
## The original population data is from https://www.cbs.nl/-/media/cbs/dossiers/nederland-regionaal/vierkanten/100/2023-cbs_vk100_2020_vol.zip (for 2020) and https://www.cbs.nl/-/media/cbs/dossiers/nederland-regionaal/vierkanten/100/2017-cbsvierkant100m.zip (for 2010)
## The original elevation data was retrieved by connecting to a Web Coverage Service (WCS) within QGIS: https://api.ellipsis-drive.com/v3/ogc/wcs/e96b10b9-e964-414c-958c-57a9dbe24e62 (AHN2) and https://service.pdok.nl/rws/ahn/wcs/v1_0 (AHN4)
## The original land use data is from PDOK https://www.pdok.nl/introductie/-/article/cbs-bestand-bodemgebruik-2010 (for 2010) and https://geodata.cbs.nl/files/Bodemgebruik/NBBG2020/ (for 2020 - 2024)
## The original administrative boundaries form the Netherlands is from PDOK and can be found in https://service.pdok.nl/kadaster/bestuurlijkegebieden/wfs/v1_0?request=GetCapabilities&service=WFS, for the project the information was retrieved via WFS request.
## The original Groundwater wells data is from PDOK and can be found https://www.pdok.nl/introductie/-/article/bro-grondwatergebruiksysteem-guf-
## The processed population, elevation data, land use, administrative boundaries and groundwater wells can be downloaded can be found in the provided zip file.
# Running the code: 
## download the processed data and update the file paths in the configuration file to match your local directory structure before execution
# Disclaimer:
## The authors declared that they used genAI to improve the code and fix bugs.