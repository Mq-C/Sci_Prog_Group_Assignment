# Assignment 1 - subsitool

# Objective:
## We are creating a python project which does a subsidence (shrinking) analysis based on elevation, the program takes two datasets for elevation (ANH2 and ANH4) and compute the difference in the elevation between them (the program is not considering the precision of both datasets). After computing the difference of the elevation, it applies a filter based on distance (50 m) and a KD-tree to order the distances and produces different clusters. The program is only selecting the top five points where the difference in the elevation is between 20 - 50 cm (important). After producing the clusters, the program using a convexhull proccess creates the polygons grouping each cluster. Finally, with the polygons the program does a analysis of the relation between the subsidence (shrinking) and five variables (populaton 2010 and 2020), land use (2010 and 2020-2024) and groundwater well (from 1900 to 2025) trying to find a relation between the variables. The analysis is presented in maps, plots and summaries througth the code. We this assignment we try to implement the concept learned from the course throught the units 1 to 5.

# Data Setup:
## The raw population data is from https://www.cbs.nl/-/media/cbs/dossiers/nederland-regionaal/vierkanten/100/2023-cbs_vk100_2020_vol.zip (for 2020) and https://www.cbs.nl/-/media/cbs/dossiers/nederland-regionaal/vierkanten/100/2017-cbsvierkant100m.zip (for 2010)
## The raw elevation data was retrieved by connecting to a Web Coverage Service (WCS) within QGIS: https://api.ellipsis-drive.com/v3/ogc/wcs/e96b10b9-e964-414c-958c-57a9dbe24e62 (AHN2) and https://service.pdok.nl/rws/ahn/wcs/v1_0 (AHN4)
## The raw land use data is from PDOK https://www.pdok.nl/introductie/-/article/cbs-bestand-bodemgebruik-2010 (for 2010) and https://geodata.cbs.nl/files/Bodemgebruik/NBBG2020/ (for 2020 - 2024)
## The raw administrative boundaries form the Netherlands is from PDOK and can be found in https://service.pdok.nl/kadaster/bestuurlijkegebieden/wfs/v1_0?request=GetCapabilities&service=WFS, for the project the information was retrieved via WFS request.
## The raw Groundwater wells data is from PDOK and can be found https://www.pdok.nl/introductie/-/article/bro-grondwatergebruiksysteem-guf-

## The processed population, elevation data, land use, administrative boundaries and groundwater wells can be downloaded can be found in the provided zip file.

# Disclaimer:
## The authors declared that they used genAI to improve the code and fix bugs.
