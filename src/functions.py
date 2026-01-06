import geopandas as gpd
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

######################################
#function to load layers and check crs
######################################
def load_layers(file_path):
    return gpd.read_file(file_path)

def check_crs(gdf):
    if gdf.crs is None:
        raise ValueError("NO CRS defined.")
    if gdf.crs.to_epsg() != 28992:
        gdf = gdf.to_crs(epsg=28992)
    return gdf

###########################################
#function for elevation difference analysis
###########################################

def point_to_xy(gdf: gpd.GeoDataFrame,value_column:str)->pd.DataFrame:

    #check if value_column(elevation value column) exists in gdf
    if value_column not in gdf.columns:
        raise KeyError(f"{value_column} column not found in GeoDataFrame.")
    
    #extract only x,y and value_column
    working_gdf =gdf[[gdf.geometry.name,value_column]].copy()

    #filter only Point with valid geometries
    is_valid_point = (working_gdf.geometry.type == 'Point') & (working_gdf.geometry.notna())
    working_gdf = working_gdf[is_valid_point]

    if working_gdf.empty:
        raise ValueError("No valid Point geometries found in GeoDataFrame.")
    
    #ensure elevation values are numeric and finite
    elev_values = pd.to_numeric(working_gdf[value_column], errors='coerce')
    is_finite = elev_values.notna() & np.isfinite(elev_values)

    #build the dataframe with x,y and elevation value
    return pd.DataFrame({
        'x': working_gdf.geometry.x[is_finite],
        'y': working_gdf.geometry.y[is_finite],
        value_column: elev_values[is_finite]
    }).reset_index(drop=True)

#The following function: bool = True/False to return as GeoDataFrame or DataFrame
def match_by_xy_and_diff(gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame, 
                         value_column: str,
                         decimals: int = 3,
                         as_geodataframe: bool = True)-> gpd.GeoDataFrame | pd.DataFrame:
    
    #convert spatial data to simple x,y,value dataframe
    points_AHN2 = point_to_xy(gdf1, value_column).rename(columns={value_column: 'elev_AHN2'})
    points_AHN4 = point_to_xy(gdf2, value_column).rename(columns={value_column: 'elev_AHN4'})

    #match points with same x,y coordinates
    matched_points = pd.merge(points_AHN2, points_AHN4, on=['x', 'y'], how='inner')

    #calculate elevation difference
    matched_points['elev_diff'] = matched_points['elev_AHN2'] - matched_points['elev_AHN4']

    num_positive = (matched_points['elev_diff'] > 0).sum()
    num_negative_stable = (matched_points['elev_diff'] <= 0).sum()
    total = len(matched_points)
    
    # Print the stats to the console
    print(f"Analysis Summary:")
    print(f"  - Rising points (Positive): {num_positive} ({(num_positive/total)*100:.1f}%)")
    print(f"  - Sinking/Stable points (Negative/Zero): {num_negative_stable} ({(num_negative_stable/total)*100:.1f}%)")

    if as_geodataframe:
        geometry = gpd.points_from_xy(matched_points['x'], matched_points['y'])
        result = gpd.GeoDataFrame(matched_points, geometry=geometry, crs="EPSG:28992")

        #add color column based on elev_diff, if elev_diff >0:red, < or =0:green
        result['color'] = 'green'
        result.loc[result['elev_diff'] > 0, 'color'] = 'red'

        fig,ax = plt.subplots(figsize=(10,10))
        result.plot(ax=ax, color=result['color'], markersize=5)
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='elev_diff <= 0', markerfacecolor='green', markersize=8),
            Line2D([0], [0], marker='o', color='w', label='elev_diff > 0', markerfacecolor='red', markersize=8)
        ]
        ax.legend(handles=legend_elements, title='Elevation Difference (m)')
        ax.set_title('Gronigen Elevation Difference between AHN2 and AHN4 Points')
        ax.set_axis_off()
        plt.show()
        return result
    return matched_points

        

 
    

    

def rename_pop_id(gdf):
    if 'C28992R100' in gdf.columns:
        return gdf.rename(columns={'C28992R100': 'grid_id'})
    
    if 'crs28992res100m' in gdf.columns:
        return gdf.rename(columns={'crs28992res100m': 'grid_id'})
    
    raise KeyError("No recognized population ID column found.")