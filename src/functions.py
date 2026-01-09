######################################
#Import the main libraries
######################################

import geopandas as gpd
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from shapely.geometry import Point
from scipy.spatial import KDTree
from collections import deque
import os
import requests
import json
import geojson
from matplotlib.cm import get_cmap
from shapely.ops import unary_union
from matplotlib.patches import Patch
import math
import fiona
from pyproj import CRS
from shapely.geometry import MultiPoint, Polygon, MultiPolygon

######################################
#function request bounary layer
######################################
# Creating a function to request Administrative boundaries via PDOK an using as a parameter a endpoint in the function an return a geojson file

 
def get_feature(endpoint):

    all_features = []
    start_index = 0
    count = 1000

    while True:
        params = {
            'request': 'GetFeature',
            'service': 'WFS',
            'version': '2.0.0',
            'typeNames': 'bestuurlijkegebieden:Gemeentegebied',
            'outputFormat': 'application/json',
            'crs': 'urn:ogc:def:crs:EPSG::28992',
            'count': count,
            'startIndex': start_index
        }
    
        response = requests.get(endpoint, params=params)
    
        if response.status_code != 200:
            print(response.text)
            break
    
        data = response.json()

        features = data["features"]                     # Extract the values stored in data under the key word 'features'.
    
        if not features:
            break

        all_features.extend(features)                   # add elementes from the previous step (features) to the end of a list.
        start_index += count

        print(f"Downloaded {len(all_features)} features")

    # build final GeoJSON
    full_geojson = {
        "type": "FeatureCollection",
        "features": all_features
            }

    with open("Administrative_boundaties.geojson", "w") as f:
        geojson.dump(full_geojson, f, indent=4)


#################################
##Request boundary layer#########
#################################

#######################################
#function to load layers and check crs#
#######################################
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

def point_to_xy(gdf: gpd.GeoDataFrame,value_column:str,decimals: int =1)->pd.DataFrame:

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
                         decimals: int = 1,
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
    print(f"  - Sinking points (Positive): {num_positive} ({(num_positive/total)*100:.1f}%)")
    print(f"  - Rising/Stable points (Negative/Zero): {num_negative_stable} ({(num_negative_stable/total)*100:.1f}%)")

    if as_geodataframe:
        geometry = gpd.points_from_xy(matched_points['x'], matched_points['y'])
        result = gpd.GeoDataFrame(matched_points, geometry=geometry, crs="EPSG:28992")

        #add color column based on elev_diff, if elev_diff >0:red, < or =0:green
        #result['color'] = 'green'
        #result.loc[result['elev_diff'] > 0, 'color'] = 'red'

        #fig,ax = plt.subplots(figsize=(10,10))
        #result.plot(ax=ax, color=result['color'], markersize=5)
        #legend_elements = [
        #    Line2D([0], [0], marker='o', color='w', label='elev_diff <= 0', markerfacecolor='green', markersize=8),
       #     Line2D([0], [0], marker='o', color='w', label='elev_diff > 0', markerfacecolor='red', markersize=8)
       # ]
       # ax.legend(handles=legend_elements, title='Elevation Difference (m)')
       # ax.set_title('Gronigen Elevation Difference between AHN2 and AHN4 Points')
       # ax.set_axis_off()
       # plt.show()
        return result
    return matched_points

#############
###KD tree###
#############
def filter_out_sinking_point(elev_diff_gdf:gpd.GeoDataFrame)->gpd.GeoDataFrame:
    #filter out sinking points where elev_diff >0
    if 'elev_diff' not in elev_diff_gdf.columns:
        raise KeyError("elev_diff column not found in GeoDataFrame.")
    
    df = elev_diff_gdf[elev_diff_gdf['elev_diff'] > 0].copy()
    print(df['elev_diff'].describe())
    return df.reset_index(drop=True)

def between_20_and_50cm_diff(elev_diff_gdf:gpd.GeoDataFrame)->gpd.GeoDataFrame:
    #considering only points with elevation difference between 20 and 50cm
    if 'elev_diff' not in elev_diff_gdf.columns:
        raise KeyError("elev_diff column not found in GeoDataFrame.")
    
    df = elev_diff_gdf[(elev_diff_gdf['elev_diff'] >= 0.2) & (elev_diff_gdf['elev_diff'] <= 0.5)].copy()
  
    return df.reset_index(drop=True)


def KD_clustering(gdf: gpd.GeoDataFrame,seed: int =43):
    df = between_20_and_50cm_diff(gdf)
    if df.empty:
        return df

    df = df.reset_index(drop=True)

    coords = np.c_[df.geometry.x,df.geometry.y]
    tree = KDTree(coords)
    radius = 50 #in meters
    n = len(coords)
    labels = -np.ones(n,dtype=int)
    cluster_id =0

    for i in range(n):
        if labels[i] != -1:
            continue

        labels[i] =cluster_id
        queue = deque([i])

        while queue:
            idx = queue.popleft()
            neighbors = tree.query_ball_point(coords[idx],r=radius)

            for nb in neighbors:
                if labels[nb] ==-1:
                    labels[nb]=cluster_id
                    queue.append(nb)
    
        cluster_id +=1
    unique_clusters = len(np.unique(labels[labels != -1]))
    print(f"Total clusters generated: {unique_clusters}")
    df['cluster'] = labels
    
    # sort by count (descending) and take the top 5
    # exclude -1 if it exists (usually noise/unassigned points)
    top_5_clusters = df[df['cluster'] != -1]['cluster'].value_counts().head(5)
    top_5_ids=top_5_clusters.index.tolist()
    top_5_df = df[df['cluster'].isin(top_5_ids)].copy()
    
    
    print("\n--- Top 5 Cluster Analysis Summary ---")
    print(f"Total unique clusters found: {len(np.unique(labels[labels != -1]))}")
    
    if top_5_clusters.empty:
        print("No clusters were generated.")
    else:
        for cid, count in top_5_clusters.items():
            print(f"Cluster {cid}: {count} points")
    print("-------------------------------------\n")

    #save top 3 clusters as geopackage
    output_path = "F:/master/mod_2/Sci_Prog_For_Geospatial_Sciences/Sci_Prog_Group_Assignment/outputs/top 5 sinking clusters.gpkg"
    
    top_5_df.to_file(output_path,driver ='GPKG',layer ='top 5 sinking clusters',mode='w')
    return df

def plot_clusters(clustered_gdf,boundary_gdf):

    if clustered_gdf is None or clustered_gdf.empty:
        print("No sinking points found to plot.")
        return

    # find the top 5 clusters by point count
    # exclude -1 (noise) to ensure only get valid clusters
    top_5_ids = clustered_gdf[clustered_gdf['cluster'] != -1]['cluster'].value_counts().head(5).index.tolist()

    if not top_5_ids:
        print("No valid clusters found to plot.")
        return

    # filter the gdf to only include those 3 clusters
    plot_gdf = clustered_gdf[clustered_gdf['cluster'].isin(top_5_ids)].copy()
    
    # plot using categorical=True for distinct cluster IDs
    fig, ax = plt.subplots(figsize=(10, 10))
    boundary_gdf.plot(ax=ax,
                  color='none',
                  edgecolor='black',
                  linewidth=1,
                  label='Groningen boundary',
                 )
    plot_gdf.plot(
        column='cluster', 
        categorical=True, 
        legend=True, 
        markersize=5, 
        cmap='tab10', #use tab10 as it's cleaner for just 5 groups
        ax=ax,
        legend_kwds={'title': "Top 5 Cluster IDs", 'bbox_to_anchor': (1.3, 1)}
    )
    
    # add Dutch RD New coordinate details
    plt.title("Top 5 Sinking Areas (20cm-50cm Range) in Groningen (AHN2 and AHN4)")
    plt.xlabel("RD Easting (meters)")
    plt.ylabel("RD Northing (meters)")
    plt.grid(False)
    
    plt.show()

#############################################
##Generating polygons top 5 clusters#########
#############################################


#############################################
##CLipping datasets to the 5 polygons########
#############################################


#############################################
##Population analysis on the top 5 clusters##
#############################################


#change population polygons id column to a common name
def clean_population_layers(gdf_input,year):
    
    gdf = gdf_input.copy()

    #rename the ID column to a common name for both population layers
    if 'C28992R100' in gdf.columns:
        gdf=gdf.rename(columns={'C28992R100': 'grid_id'})
    
    elif 'crs28992res100m' in gdf.columns:
        gdf=gdf.rename(columns={'crs28992res100m': 'grid_id'})
    else:
        raise KeyError("No recognized population ID column found.")

    #rename the population count layer to a common name
    if year ==2010:
        if 'INW2010' in gdf.columns:
            gdf = gdf.rename(columns={'INW2010':'pop_count'})
    elif year == 2020:
        if 'aantal_inwoners' in gdf.columns:
            gdf = gdf.rename(columns={'aantal_inwoners':'pop_count'})

    #filter out no data '-9998'
    no_data_value = [-99998,-99997]
    gdf = gdf[~gdf['pop_count'].isin(no_data_value)].copy()

    return gdf[['grid_id','pop_count','geometry']]

def population_analysis(pop_2010_gdf,pop_2020_gdf,cluster_poly):
    pop_2010 = clean_population_layers(pop_2010_gdf,2010)
    pop_2020 = clean_population_layers(pop_2020_gdf,2020)

    matched_pop_df = pd.merge(
        pop_2010,
        pop_2020.drop(columns='geometry'),
        on='grid_id',
        suffixes=('_2010','_2020')
    )

    matched_pop_gdf = gpd.GeoDataFrame(matched_pop_df,geometry='geometry',crs='EPSG:28992')

    print(f'Total population polygons matched between 2010 and 2020: {len(matched_pop_df)}')

    #clip population polygon to the 5 cluster
    clipped_pop = gpd.sjoin(matched_pop_df,
                            cluster_poly[['cluster_id','geometry']],
                            how='inner',
                            predicate='intersects').copy()
    
    print(f'Totlal matched popualtion polygons  within the 5 clusters: {len(clipped_pop)}')

    #calculate difference
    clipped_pop['pop_diff'] =(clipped_pop['pop_count_2020']-clipped_pop['pop_count_2010'])

    clipped_pop['trend']=np.select(
        [clipped_pop['pop_diff']>0,clipped_pop['pop_diff']<0,clipped_pop['pop_diff']==0],
        ['Increased','Decreased','No change'],
        default='No change'
    )
    summary =clipped_pop.groupby(['cluster_id','trend']).size().unstack(fill_value=0)
    print('---Population trend summary by cluster id')
    print(summary)


    return clipped_pop,summary






