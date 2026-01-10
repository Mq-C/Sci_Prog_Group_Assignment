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
#function request boundary layer
######################################
### Function to request  via WFS in PDOK the Administrative boundaries for the Netherlands.
### It takes as input parameter the endpoint a return a return a geojson file.

 
def get_feature(endpoint):
    '''
    The function request via WFS and endpoint the administrative boundaries for the Netherlands considering a series of parameters.
    
    parameters: 
        endpoint (str): link to the request.
    
    Print:
        print (int): It prints a code if the reponse is different of 200.
        print (str): It prints the number of the features downloaded
        
    Returns:
        Administrative_boundaties (geojson): It contains the GeoJSON file with all the information stored
        gdf (GeoDataFrame): It contains a GeoDataFrame with the information and set a CRS.
            
    Raises: 
       RuntimeError: It will raise, if the response code is different of 200.
        ValueError: It will raise, if the proccess does not return features.
    '''
    
    # set the initial elements for the function
    all_features = []
    start_index = 0
    count = 1000

    while True:
        # Define the parameters for the request
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

        # Execute the request
        response = requests.get(endpoint, params=params)
    
        if response.status_code != 200:
            raise RuntimeError(f"WFS request failed (HTTP {response.status_code}): {response.text}")
                    
        # assign to data json files
        data = response.json()

        if 'features' not in data:
            raise ValueError('Invalid WFS response: features key not found. ')
        
        # Extract the values stored in data under the key word 'features'.
        features = data["features"]                     
    
        if not features:
            break
        # add elementes from the previous step (features) to the end of a list.
        all_features.extend(features)                   
        start_index += count

        print(f"Downloaded {len(all_features)} features")

    # build final GeoJSON file
    full_geojson = {
        "type": "FeatureCollection",
        "features": all_features
            }
    # Store the GeoJSON file considering a default name
    with open("Administrative_boundaties.geojson", "w") as f:
        geojson.dump(full_geojson, f, indent=4)
    
    # convert also in a geodataframe
    gdf = gpd.GeoDataFrame.from_features(full_geojson,crs="EPSG:28992")
    return gdf

#######################################
#function to load layers and check crs#
#######################################
#def load_layers(file_path):
    #return gpd.read_file(file_path)
def load_layers(file_path, layer=None):
    '''
    The function reads a geopackage or geojson and convert it into a GeoDataFrame.
    
    parameters: 
        path (str): Path to the geopackage file.
        layer (str): If the geopackage has different layers, specify the layer to open, otherwise layer = None.
    
    Print:
        print(gdf_head): It prints the first five rows for the geodataframe as a reference.
        
    Returns:
        gdf (GeoDataFrame): It contains the GEoDataFrame with all the information in the geopackage or geojson file.
        gdf_head (GeoDataFrame): It contains the first rows in the gdf, they are ready to print. 
    
    Raises: 
        FileNotFoundError: It will raise, if the path does not exist or it is wrong.
        ValueError: It will raise, if the file cannot be read by geopandas because the name of the layers.
    '''
    
    if not os.path.exists(file_path):
            raise FileNotFoundError(f'The file does not exist or the path is wrong:{file_path}')
    
    if file_path.lower().endswith(".gpkg"):
        layers = fiona.listlayers(file_path)
        
        if layer is None:
             raise ValueError(f'geopackage contains multiple layers: {layers}, specify the layer to read')
        
        if layer not in layers:
            raise ValueError(f'layer {layer} not found. Available leyers:{layers}')
        
    try:    
        gdf = gpd.read_file(file_path)
        gdf_head = gdf.head()
        
    except Exception as e:
        # Raise a error if for any reason geopandas cannot read the geopackage
        raise ValueError(f'Error reading the file {file_path}: {e}')
    
    #print(gdf_head)
    return gdf, gdf_head

# Function to validate the CRS project (28992) in a GeoDataFrame and if the CRS is not that, it will be reprojected to this CRS project.

#def check_crs(gdf):
    #if gdf.crs is None:
        #raise ValueError("NO CRS defined.")
    #if gdf.crs.to_epsg() != 28992:
        #gdf = gdf.to_crs(epsg=28992)
    #return gdf

def check_crs(gdf):
    '''
    The function reads a GeoDataFrame and considerer a predefine CRS based on EPSG for the project then it reads the current crs and if 
    this is different from the defined parameter, it reproject the geodataframe.
    
    parameters: 
        gdf (geodataframe): GeoDataFrame to validate and set the coordinate system
    
    Print:
        print(crs_initial): It prints the original CRS for the geodataframe.
        print(crs_final): It prints the final CRS for the geodataframe after verification.
        
    Returns:
        gdf (GeoDataFrame): It contains the GeoDataFrame with the CRS setup for the project after verification.
            
    Raises: 
        ValueError: It will raise, if the GeoDataFrame does not have a CRS defined.
    '''
    
    if gdf.crs is None:
        raise ValueError("NO CRS defined.")
    crs_initial = gdf.crs
    
    # Set the CRS target for the project
    target_crs = CRS.from_epsg(28992)
    
    # Set the CRS for the project in case it is different to the target.
    if gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)    # I suggest this improvement
        
    print(f'Initial CRS: {crs_initial}') 
    print(f'Final CRS: {gdf.crs}') 
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
    print("\n--- Full Dataset Elevation Difference Statistics ---")
    print(matched_points['elev_diff'].describe())
    print("----------------------------------------------------\n")

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
        #the following lines are for plotting the elevation change map, the result is saved. If needed,uncomment these lines.
        ##result['color'] = 'green'
        ##result.loc[result['elev_diff'] > 0, 'color'] = 'red'

        ##fig,ax = plt.subplots(figsize=(10,10))
        ##result.plot(ax=ax, color=result['color'], markersize=5)
        ##legend_elements = [
           ## Line2D([0], [0], marker='o', color='w', label='elev_diff <= 0', markerfacecolor='green', markersize=8),
           ##Line2D([0], [0], marker='o', color='w', label='elev_diff > 0', markerfacecolor='red', markersize=8)
        ##]
        ##ax.legend(handles=legend_elements, title='Elevation Difference (m)')
        ##ax.set_title('Gronigen Elevation Difference between AHN2 and AHN4 Points')
        ##ax.set_axis_off()
        ##plt.show()
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
    return top_5_df

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

### Function to create polygons based on convexhull proccess based on geodataframe which contains a list of points representing clusters,
### The function returns a geodataframe

def polygon_clusters(gdf):
    '''
    The function reads one GeoDataFrames representing clusters and based on convexhull process create a polygon for each cluster.
    The function return a geodataframe.
    
    parameters: 
        gdf (geodataframe): GeoDataFrame to create the polygons.
                            
    Print:
        print(-): This function does not print results.
                        
    Returns:
         hulls_gdf (geodataframe): It contains the GeoDataFrame with the polygons for each cluster defined.
            
    Raises: 
        TypeError: It will raise, if the input is not a GeoDataFrame.
    ''' 
    # Validation
    if not isinstance(gdf, gpd.GeoDataFrame):
            raise TypeError("Input must be a valid GeoDataFrame")
        
    # List to store cluster hulls
    hulls = []
    
    # Preprocessing step setting the geometry for each row representing points
    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.apply(lambda g: g.geoms[0] if g.geom_type == "MultiPoint" else g)
    gdf = gdf.set_geometry("geometry")   
    
    # Group points by cluster and filling a list
    for cluster_value, group in gdf.groupby('cluster', observed=True):   
        # Convert cluster points to MultiPoint
        points = MultiPoint(group.geometry.values)
        
        # Compute convex hull
        hull = points.convex_hull
        
        # Append as a dictionary
        hulls.append({'cluster_id': cluster_value, 'geometry': hull})
    
    # convert hulls to a GeoDataFrame
    hulls_gdf = gpd.GeoDataFrame(hulls, crs=gdf.crs)
    return hulls_gdf

#############################################
##CLipping datasets to the 5 polygons########
#############################################

### Function takes one geodataframe and clipped based on another geodataframe in this case (Province), and return the geodataframe clipped to the selected province.

def clip_pol(gdf1, gdf2):
    '''
    The function reads two GeoDataFrames and clipped one based on another, the function return the clipped GeoDataFrame.
    Also, the function has another function inside to fix the geometries in case of topological errors.
    
    parameters: 
        gdf1 (geodataframe): GeoDataFrame to which the clipping function will be apply.
        gdf2 (geodataframe): GeoDataFrame to apply the clipping function - overlay.
            
    Print:
        print(-): This function does not print results.
                
    Returns:
        gdf (GeoDataFrame): It contains the GeoDataFrame clipped based on the extension of the second GeoDataFrame.
            
    Raises: 
        ValueError: It will raise, if the GeoDataFrame has some null/none values, empties geometries.
        TypeError: It will raise, if the inputs are not a GeoDataFrame.
    '''  
    
    # Validation
    if not isinstance(gdf1, gpd.GeoDataFrame) or not isinstance(gdf2, gpd.GeoDataFrame):
        raise TypeError("Input must be a valid GeoDataFrame")
    
    if not gdf1.geom_type.isin(['Point', 'MultiPoint']).all():
        if gdf1.geometry.is_empty.any():
            raise ValueError("gdf1 has some empty geometries")
    else:
        pass

    if gdf1.geometry.isnull().any() or gdf2.geometry.isnull().any():
        raise ValueError("The gdf has some geometries null / None")
    
    if gdf2.geometry.is_empty.any():
        raise ValueError("The gdf has some geometries empties")
    
    # Auxiliary function to fix invalid geometries
    def fix_geometry(geom):
        if geom.is_valid:
            return geom
        try:
            return geom.buffer(0)
        except Exception:
            return geom  
        
    # Applying the fix validation geometry to the GeoDataFrames
    gdf1["geometry"] = gdf1["geometry"].apply(fix_geometry)
    gdf2["geometry"] = gdf2["geometry"].apply(fix_geometry)

    # Clipping polygons using (polygons, Province == Groningen)
    gdf = gpd.clip(gdf1, gdf2)
    return gdf

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
    joined_pop = gpd.sjoin(matched_pop_df,
                            cluster_poly[['cluster_id','geometry']],
                            how='inner',
                            predicate='intersects').copy()
    clipped_pop = joined_pop.drop_duplicates(subset='grid_id').copy()
    print(f'Totlal matched popualtion polygons  within the 5 clusters: {len(clipped_pop)}')

    #calculate difference
    clipped_pop['pop_diff'] =(clipped_pop['pop_count_2020']-clipped_pop['pop_count_2010'])

    clipped_pop['trend']=np.select(
        [clipped_pop['pop_diff']>0,clipped_pop['pop_diff']<0,clipped_pop['pop_diff']==0],
        ['Increased','Decreased','No change'],
        default='No change'
    )



    summary =clipped_pop.groupby(['cluster_id','trend']).size().unstack(fill_value=0)
    print('---Population trend summary by cluster id---')
    print(summary)


    return clipped_pop,summary


#############################################
##Filtering the Province boundary##
#############################################
### Function to filter the GeoDataFrame administrative boundary only for the chosen province (Groningen)

def filter_province(gdf, province):
    '''
    The function reads a GeoDataFrame and province name choosen and filter the data only for the data inside the province.
    
    parameters: 
        gdf (geodataframe): GeoDataFrame to apply the filter.
        province (str): Name of the province for the analysis. 
    
    Print:
        print(the number of records): It prints the number of the records after applying the function.
                
    Returns:
        gdf (GeoDataFrame): It contains the GeoDataFrame with the data after applying the function.
            
    Raises: 
        ValueError: It will raise, if the province name is no in the GeoDataFrame.
        TypeError: It will raise, if the input is not a GeoDataFrame.
    '''
    
    if  not (gdf["ligtInProvincieNaam"].str.lower() == province).any():
        raise ValueError(f'The province: {province}, is not in the geodataframe')
     
    elif not isinstance(gdf, gpd.GeoDataFrame):
        raise TypeError("Input must be a valid GeoDataFrame")
    
    else:
        gdf = (gdf[gdf["ligtInProvincieNaam"].str.lower() == province])
        print(f'the number of records: {len(gdf)}')
        return gdf

#### Function to merge the polygons for Groningen into one polygon representing all the province.

def merg_province(gdf):
    '''
    The function reads a GeoDataFrame and merge all the polygons inside into one unique polygon representing the chosen province.
    This function has another function inside to fix the geometry in case the polygons have some topological errors.
    
    parameters: 
        gdf (geodataframe): GeoDataFrame to apply the merge.
            
    Print:
        print(-): This function does not print results.
                
    Returns:
        gdf (GeoDataFrame): It contains the GeoDataFrame with the data after merging the polygons.
            
    Raises: 
        ValueError: It will raise, if the GeoDataFrame has some null/none values, empties geometries.
        TypeError: It will raise, if the input is not a GeoDataFrame.
    '''
    
    # Validation of the GeoDataFrame
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise TypeError("Input must be a valid GeoDataFrame")
    
    if gdf.geometry.isnull().any():
        raise ValueError("The gdf has some geometries null / None")
    
    if gdf.geometry.is_empty.any():
        raise ValueError("The gdf has some geometries empties")
    
    if not gdf.is_valid.all():
        # Auxiliary function to fix invalid geometries
        def fix_geometry(geom):
            if geom is None or geom.is_empty:
                return geom
            try:
                return geom.buffer(0)  
            except Exception:
                return geom
        # Applying the merge function
        gdf["geometry"] = gdf["geometry"].apply(fix_geometry)
        Unary_union = unary_union(gdf.geometry)
        gdf= gpd.GeoDataFrame(index=[0], crs=gdf.crs, geometry=[Unary_union], data={"ligtInProvincieNaam": ["Groningen"]})
        return gdf
   
    else:
        # Applying the merge function
        Unary_union = unary_union(gdf.geometry)
        gdf= gpd.GeoDataFrame(index=[0], crs=gdf.crs, geometry=[Unary_union], data={"ligtInProvincieNaam": ["Groningen"]})
        return gdf

#############################################
##Computing the area in Ha##
#############################################

### Function which takes a GeoDataFrame representing polygons and returns the area in Hectares (Ha)

def area_ha(gdf):
    '''
    The function reads one GeoDataFrames and compute the area in Hectares (Ha), and it rounds the output to three digits.
    Also, the function has another function inside to fix the geometries in case of topological errors.
    
    parameters: 
        gdf (geodataframe): GeoDataFrame to apply the area calculation.
                    
    Print:
        print(-): This function does not print results.
                
    Returns:
        gdf (GeoDataFrame): It contains the GeoDataFrame with a new column called area_ha with the output.
            
    Raises: 
        ValueError: It will raise, if the GeoDataFrame has some null/none values, empties geometries.
        TypeError: It will raise, if the input is not a GeoDataFrame.
    ''' 
    # Validation
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise TypeError("Input must be a valid GeoDataFrame")
    
    if gdf.geometry.isnull().any():
        raise ValueError("The gdf has some geometries null / None")
    
    if gdf.geometry.is_empty.any():
        raise ValueError("The gdf has some geometries empties")
    
    if not gdf.is_valid.all():
        # Auxiliary function to fix invalid geometries
        def fix_geometry(geom):
            if geom is None or geom.is_empty:
                return geom
            try:
                return geom.buffer(0) 
            except Exception:
                return geom
        # Applying the fix validation geometry to the GeoDataFrames
        gdf["geometry"] = gdf["geometry"].apply(fix_geometry)
        # Computing the area in Ha for each row in the GeoDataFrame
        gdf['area_ha'] = gdf.geometry.area / 10000
        # Lambda function to round the results
        gdf['area_ha'] = gdf['area_ha'].apply(lambda x: round(x, 3))
        return gdf
    else:
        # Computing the area in Ha for each row in the GeoDataFrame
        gdf['area_ha'] = gdf.geometry.area / 10000
        # Lambda function to round the results
        gdf['area_ha'] = gdf['area_ha'].apply(lambda x: round(x, 3))
        return gdf

#############################################
##Land use analysis for 2010 and 2020##
#############################################
### Function which evaluates if a column exist in a Geodataframe. If the column exists, it will create a new field and fill with information (specially for the land use 2020)

def create_field_pol(gdf, name_col):
        '''
    The function reads one GeoDataFrames and based on name_col, it evaluates if the name_col exists, and if it is true, it will create a 
    new column and fills the column with a names based on codes published in PDOK website (processing data step).
    
    parameters: 
        gdf (geodataframe): GeoDataFrame to apply the creation of a new field.
        name_col (str): name of the column target to apply the function.
                    
    Print:
        print(-): This function does not print results.
                
    Returns:
        gdf (GeoDataFrame): It contains the GeoDataFrame with a new column categorie with the output.
            
    Raises: 
        ValueError: It will raise, if the GeoDataFrame has some null/none values, empties geometries.
        TypeError: It will raise, if the input is not a GeoDataFrame.
    ''' 
        # Validation
        if not isinstance(gdf, gpd.GeoDataFrame):
                raise TypeError("Input must be a valid GeoDataFrame")
        
        if not isinstance(name_col, str):
                raise TypeError(f"name_col must be a string, got {type(name_col).__name__}")
        
        if gdf.geometry.is_empty.any():
                raise ValueError("The gdf has some geometries empties")

        # Filling the field categorie based on codes (the codes are specified in PDOK website)
        cols_lower = {c.lower(): c for c in gdf.columns}
        if name_col.lower() in cols_lower:
                real_col = cols_lower[name_col.lower()]
                NBBG20_key_list = sorted(gdf[real_col].unique().tolist())
                NBBG20_values_list = ['Spoorterrein', 'Hoofdweg', 'Vliegveld', 'Woonterrein', 'Detailhandel en horeca', 'Openbare voorziening', 'Sociaal-culturele voorziening', 'Bedrijfsterrein', 'Stortplaats', 'Begraafplaats', 'Delfstofwinplaats', 'Bouwterrein', 'Semi-verhard overig terrein', 'Zonnepark', 'Park en plantsoen', 'Sportterrein', 'Volkstuin', 'Dagrecreatief terrein', 'Verblijfsrecreatief terrein', 'Glastuinbouw', 'Overig agrarisch terrein', 'Akker en meerjarige teelt', 'Agrarisch grasland', 'Bos', 'Open droog natuurlijk terrein', 'Open nat natuurlijk terrein', 'Natuurlijk grasland', 'Kustduinen', 'Afgesloten zeearm', 'Recreatief binnenwater', 'Binnenwater voor delfstofwinning', 'Vloei- en/of slibveld', 'Overig binnenwater', 'Waddenzee, Eems & Dollard', 'Noordzee']
                NBBG20_Map = dict(zip(NBBG20_key_list, NBBG20_values_list))
                gdf['categorie'] = gdf['NBBG20'].map(NBBG20_Map)
                return gdf
        else:
                return gdf
            
            
#### Function which takes one geodataframe and based on specific categorie plotted the data, and put as a title the input titles. This functions is useful for plotting
### land use for 2010 and 2020

# Plotting a map for land use 2010 and 2020 for Groningen Province and in the final part print summaries

def plot_classes(gdf, categorie, titles):
    '''
    The function reads one GeoDataFrames and based on a specific column (categorie) plotted a map with a specific legend and title set.
    
    parameters: 
        gdf (geodataframe): GeoDataFrame to plot a map.
        categorie (str): name of the column which will be used to plot the map.
        titles (str): the the main title name and title legend.
                            
    Print:
        print(Title): It prints the title of the map as a reference.
        print(Total categories): It prints the number of the classes inside the map.
        print(Title): It prints the title of the map as a reference.
        print(Summary): It prints the classes inside the map grouped and the total sum of the area in Ha for each group.
                
    Returns:
        plot (plot): It contains the map with the categories.
        
    Raises: 
        TypeError: It will raise, if the input is not a GeoDataFrame or the inputs categorie and titles are not strings.
    '''
        
    # Validation
    if not isinstance(gdf, gpd.GeoDataFrame):
            raise TypeError("Input must be a valid GeoDataFrame")
        
    if not isinstance(categorie, str) or not isinstance(titles, str):
                raise TypeError(f"categorie must be a string, got {type(categorie).__name__},titles must be a string, got{type(titles).__name__}")
    
    # Creating a list for the legend, the colors for plotting
    Land_uses_legend = sorted(gdf[categorie].dropna().unique())
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i % cmap.N) for i in range(len(Land_uses_legend))]
    category_colors = dict(zip(Land_uses_legend, colors))
    gdf['color'] = gdf[categorie].map(category_colors).fillna('lightgrey')
    # Plotting the geodataframe and setting the variables for plotting
    fig, ax = plt.subplots(figsize=(14,8))
    gdf.plot(ax=ax, color=gdf['color'], edgecolor="black", linewidth=0.3)
    ax.set_title(titles, fontsize=16, fontweight='bold', pad=12)
    ax.set_xlabel('X', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y', fontsize=12, fontweight='bold', rotation=0)
    handles = [Patch(facecolor=category_colors[cat], edgecolor='black', label=cat) 
           for cat in Land_uses_legend]
    leg=ax.legend(handles=handles,
            loc = 'lower center',
            bbox_to_anchor = (0.5, -0.40),
            ncol=math.ceil(len(Land_uses_legend)/6),
            frameon = False,
            fontsize = 10, 
            title = titles)
    leg.get_title().set_fontweight('bold')
    plt.tight_layout()
    plt.show()
    # Printing some summaries for comparison between datasets
    print(f'Summary for: {titles}\n')
    print(f"Total categories: {len(gdf)}\n")
    summary = (gdf.groupby(categorie)['area_ha'].sum().sort_values(ascending=False))
    print('Categories grouped by area in Ha\n')
    print(summary)
    
#### Function which takes one geodataframe and based on specific categorie create a bar chart, and put as a title the input titles.

# Plotting a bar chart for land use 2010 and 2020 for Groningen Province 

def plot_bar_classes(gdf, categorie, titles):
    '''
    The function reads one GeoDataFrames and based on specific column name (categories), plotted a bar chart grouping the sum of the 
    area for each categories, the function also takes as an argument titles to assign the title to the plot.
    
    parameters: 
        gdf (geodataframe): GeoDataFrame to plot a bar chart.
        categorie (str): column name to plot in the bar chart.
        titles (str): name to put in the bar chart.
                    
    Print:
        print(-): This function does not print results.
                        
    Returns:
         plot (bar chart): This function plots a bar chart grouped by categories.
            
    Raises: 
        TypeError: It will raise, if the input is not a GeoDataFrame.
    ''' 
    # Validation
    if not isinstance(gdf, gpd.GeoDataFrame):
            raise TypeError("Input must be a valid GeoDataFrame")
        
    if not isinstance(titles, str) or not isinstance(categorie, str):
                raise TypeError(f"categorie must be a string, got {type(categorie).__name__}, titles must be a string, got{type(titles).__name__}")
        
    # create a list of unique values for categorie in order
    Land_uses_legend = sorted(gdf[categorie].dropna().unique())
    
    # For each categorie sum the total area in Ha and set the function to assign a color, category_color y bar_color to use to create the bar chart.
    area_land_use_class = (gdf.groupby(categorie, as_index=False)['area_ha'].sum())
    area_land_use_class[categorie] = pd.Categorical(area_land_use_class[categorie], categories=Land_uses_legend, ordered=True)
    area_land_use_class = area_land_use_class.sort_values(categorie)
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i % cmap.N) for i in range(len(Land_uses_legend))]
    category_colors = dict(zip(Land_uses_legend, colors))
    bar_colors = [category_colors[c] for c in area_land_use_class[categorie]]
    
    # Plotting the categories in a bar chart and setting the plot format
    plt.figure(figsize=(8,5))
    plt.bar(area_land_use_class[categorie], area_land_use_class['area_ha'], color = bar_colors)
    plt.xlabel(categorie)
    plt.ylabel('Total area Ha')
    plt.title(f'{titles} Province total area by categorie')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    

#####################################################
##Groundwater well analysis for 2010 and 2010-2021##
#####################################################

### Function which evaluate if a specific value exist in a column of a geodataframe, and if the value exists, then the name is replaced by another name (specially for groundawater wells)

def update_field(gdf, column_name, original_name, final_name):
        '''
    The function reads one GeoDataFrames and based on column_name, it evaluates if the column_name exists, and if it is true, it will search for original_name, and if
    the orinal_name is found, it will be updated for the final_name (processing data step).
    
    parameters: 
        gdf (geodataframe): GeoDataFrame to apply the update field.
        column_name (str): name of the column target to apply the function.
        original_name (str): name of value target to apply the function.
        final_name (str): final name after applying the function.
                    
    Print:
        print(-): This function does not print results.
                
    Returns:
        gdf (GeoDataFrame): It contains the GeoDataFrame with the values in the column_name updated.
            
    Raises: 
        TypeError: It will raise, if the input is not a GeoDataFrame or the inputs column_name, original_name, final_name are not strings.
    ''' 
        # Validation
        if not isinstance(gdf, gpd.GeoDataFrame):
                raise TypeError("Input must be a valid GeoDataFrame")
        
        if not isinstance(original_name, str) or not isinstance(final_name, str):
                raise TypeError(f"original_name must be a string, got {type(original_name).__name__},original_name must be a string, got{type(final_name).__name__}")
        
        # Updating the valued name based on the condition
        cols_lower = {c.lower(): c for c in gdf.columns}
        if column_name.lower() in cols_lower:
                real_col = cols_lower[column_name.lower()]
                values_lower = {str(v).lower(): v for v in gdf[real_col].dropna().unique()}
                if original_name.lower() in values_lower:
                        real_value = values_lower[original_name.lower()]
                        gdf[real_col]=gdf[real_col].replace(real_value, final_name)
                return gdf
        else:
                return gdf
            
### Function which takes a geodataframe a name_col, it creates a new field based on name_col and fill it grouping the data by year wells in three intervals
### <=2010', '2011-2022', '>2022', these intevals are related to the elevation data sets

def create_field_pt(gdf, name_col):
        '''
    The function reads one GeoDataFrames and creates a new column (name_col). The function takes a specific columns and coverts it
    to data time format, and after that it groupes the data based on date intervale previously defined (processing data step).
    
    parameters: 
        gdf (geodataframe): GeoDataFrame to apply the field creation.
        name_col (str): name of the new column which will be filled after applying the function.
                            
    Print:
        print(-): This function does not print results.
                
    Returns:
        gdf (GeoDataFrame): It contains the GeoDataFrame with the values in the name_col updated.
            
    Raises: 
        TypeError: It will raise, if the input is not a GeoDataFrame or the input name_col is not string.
    '''
        
        # Validation
        if not isinstance(gdf, gpd.GeoDataFrame):
                raise TypeError("Input must be a valid GeoDataFrame")
        
        if not isinstance(name_col, str):
                raise TypeError(f"name_col must be a string, got {type(name_col).__name__}")
        
        # Converting the column name to datatime format to manipulate the data
        gdf['brogufvolledigeset — lifespan_start_time']= pd.to_datetime(gdf['brogufvolledigeset — lifespan_start_time'], errors='coerce')
        # Filling the field name_col based on dates ranges
        gdf[name_col] = pd.cut(gdf['brogufvolledigeset — lifespan_start_time'].dt.year, bins=[0,2010,2022,2100], labels=['<=2010', '2011-2022', '>2022'], right=True)
        return gdf
    
#### Function which takes one geodataframe and based on specific categorie plotted the data, and put as a title the input title. This functions is useful for plotting
### groundwater wells and overlay an based layer specificed and another overlay layers specified.

# Plotting a map for Wells for Groningen Province and in the final part print summaries.

def plot_classes_point(gdf, categorie, titles, base_layer=None, overlay_layers=None):
    '''
    The function reads one GeoDataFrames related to points and based on a specific column (categorie) plotted a map with a specific legend and title set.
    Also, the function can take a base_layer and overlay_layers to improve the plot.
    
    parameters: 
        gdf (geodataframe): GeoDataFrame to plot a map.
        categorie (str): name of the column which will be used to plot the map.
        titles (str): the the main title name and title legend.
        base_layer (geodataframe): Additional GeoDataFrame to plot on the map.
        overlay_layers (geodataframe): Additional GeoDataFrame to plot on the map.
                            
    Print:
        print(Title): It prints the title of the map as a reference.
        print(Total categories): It prints the number of the classes inside the map based on specific column.
        print(Title): It prints the title of the map as a reference.
        print(Summary): It prints the classes inside the map grouped and the total sum of the area in Ha for each group.
                
    Returns:
        plot (plot): It contains the map with the categories.
        ax (object): It contains the object related to the plot
        
    Raises: 
        TypeError: It will raise, if one of the inputs are not a GeoDataFrame or the inputs categorie and titles are not strings.
    '''
    # Validation
    
    if not isinstance(gdf, gpd.GeoDataFrame):
            raise TypeError("Input must be a valid GeoDataFrame")
    
    if base_layer is None or overlay_layers is None:
        pass
    else:
        if not isinstance(base_layer, gpd.GeoDataFrame) or not isinstance(overlay_layers, gpd.GeoDataFrame):
            raise TypeError("Inputs must be a valid GeoDataFrame")
        
    if not isinstance(categorie, str) or not isinstance(titles, str):
                raise TypeError(f"categorie must be a string, got {type(categorie).__name__},titles must be a string, got{type(titles).__name__}")
            
    # Copy and flatten the index
    gdf_plot = gdf.copy().reset_index(drop=True)

    # Filtering out missing values
    gdf_plot = gdf_plot[gdf_plot[categorie].notna()]

    # Convert categorical to string temporarily for mapping
    gdf_plot['temp_category'] = gdf_plot[categorie].astype(str)

    # Get the categories in order if categorical
    if isinstance(gdf[categorie].dtype, pd.CategoricalDtype):
        categories = list(gdf[categorie].cat.categories)
    else:
        categories = sorted(gdf_plot['temp_category'].unique())

    # Keep only categories present in the data
    categories_present = [cat for cat in categories if cat in gdf_plot['temp_category'].values]

    # Dynamic colors
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i / len(categories_present)) for i in range(len(categories_present))]
    category_colors = dict(zip(categories_present, colors))

    # Map colors using the string column
    gdf_plot['color'] = gdf_plot['temp_category'].map(category_colors)

    # Plotting 
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_aspect('equal')
    if base_layer is not None:
        base_layer.plot(ax=ax, edgecolor='green', facecolor='lightyellow',aspect = 'equal')
    
    if overlay_layers is not None:
        overlay_layers.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.3,aspect = 'equal')
    
    # Detect geometry type
    geom_type = gdf_plot.geometry.geom_type.unique()
    if all(gt in ['Point', 'MultiPoint'] for gt in geom_type):
        gdf_plot.plot(ax=ax, color=gdf_plot['color'], edgecolor='black', markersize=50,aspect = 'equal')
    else:
        gdf_plot.plot(ax=ax, color=gdf_plot['color'], edgecolor='black', linewidth=0.3,aspect = 'equal')

    # Titles and labels
    ax.set_title(titles, fontsize=16, fontweight='bold', pad=12)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)

    # Legend
    handles = [Patch(facecolor=category_colors[cat], edgecolor='black', label=cat)
               for cat in categories_present]
    leg = ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.01, 0.95),
                    ncol=1, frameon=False, fontsize=10, title=titles)
    leg.get_title().set_fontweight('bold')

    plt.tight_layout()
    plt.show()
    
    # Summary
    print(f"Summary for: {titles}\n")
    print(f"Total categories present: {len(categories_present)}\n")
    summary = gdf_plot.groupby('temp_category', observed=True).size().sort_values(ascending=False)
    print("Categories grouped by count:\n")
    print(summary)
    return ax

#### Function which takes one geodataframe representing points and based on specific values and settings plot a bar chart.

# Plotting a bar chart for Groundwater well in Groningen Province

def plot_bar_classes_point(gdf):
    '''
    The function reads one GeoDataFrames representing points and based on specific column name (date_invertal), plotted a bar chart counting the values for  
    each category.
    
    parameters: 
        gdf (geodataframe): GeoDataFrame to plot a bar chart.
                            
    Print:
        print(-): This function does not print results.
                        
    Returns:
         plot (bar chart): This function plots a bar chart grouped by categories.
            
    Raises: 
        TypeError: It will raise, if the input is not a GeoDataFrame.
    ''' 
    # Validation
    if not isinstance(gdf, gpd.GeoDataFrame):
            raise TypeError("Input must be a valid GeoDataFrame")
        
    #creating a list grouping by date_interval and counting the number of values in each category
    wells_date = (gdf.groupby('date_interval', as_index=False, observed=True).count().sort_values(by='geometry', ascending=True))
    
    #Setting some paratemer for plotting
    n_bars = len(wells_date)
    cmap = plt.get_cmap('tab20')
    bar_colors = [cmap(i / n_bars) for i in range(n_bars)]
    
    #Plotting a bar chart
    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(wells_date['date_interval'], wells_date['geometry'], color = bar_colors)
    ax.set_xlabel('Starting time date')
    ax.set_ylabel('Total number of wells by starting date')
    ax.set_title('Total number of wells for type for Groningen Province gruped by date', fontsize=16, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.show()