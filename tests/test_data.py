import geopandas as gpd
import pandas as pd
import pytest
import requests
import matplotlib
matplotlib.use("Agg")
from shapely.geometry import Point, Polygon
from src.functions import (load_layers, 
                           check_crs,
                           match_by_xy_and_diff,
                           get_feature,
                           polygon_clusters,
                           clip_pol,
                           filter_province,
                           merg_province,
                           area_ha,
                           create_field_pol,
                           plot_classes,
                           plot_bar_classes,
                           update_field,
                           create_field_pt,
                           plot_classes_point,
                           plot_bar_classes_point,
                           clean_population_layers,
                           )
import numpy as np




elevation_AHN2_path = "F:/master/mod_2/Sci_Prog_For_Geospatial_Sciences/Pair assignment/data/AHN2_Groningen_Points.gpkg"
elevation_AHN4_path = "F:/master/mod_2/Sci_Prog_For_Geospatial_Sciences/Pair assignment/data/AHN4_Groningen_Points.gpkg"
population_2010_path = "F:/master/mod_2/Sci_Prog_For_Geospatial_Sciences/Pair assignment/data/Pop_2010.gpkg"
population_2020_path = "F:/master/mod_2/Sci_Prog_For_Geospatial_Sciences/Pair assignment/data/Pop_2020.gpkg"
#############################################
#Test for loading and checking CRS of layers#
#############################################
def test_layers_load():

    AHN2_points, _ = load_layers(elevation_AHN2_path,layer="AHN2_Groningen_Points")
    AHN4_points, _ = load_layers(elevation_AHN4_path,layer="AHN4_Groningen_Points")
    pop_2010, _  = load_layers(population_2010_path,layer= 'Pop_2010')
    pop_2020, _  = load_layers(population_2020_path,layer='Pop_2020')

    assert isinstance(AHN2_points, gpd.GeoDataFrame)
    assert isinstance(AHN4_points, gpd.GeoDataFrame)
    assert isinstance(pop_2010, gpd.GeoDataFrame)
    assert isinstance(pop_2020, gpd.GeoDataFrame)

    assert not AHN2_points.empty
    assert not AHN4_points.empty
    assert not pop_2010.empty
    assert not pop_2020.empty

def test_crs():
    #if crs is not 28992, it should be converted to 28992
    gdf = gpd.GeoDataFrame(geometry=[Point(1, 1)], crs="EPSG:4326")
    result = check_crs(gdf)
    assert result.crs.to_epsg() == 28992

def test_if_crs_already_28992():
    #if crs is already 28992, stay unchanged
    gdf = gpd.GeoDataFrame(geometry=[Point(1, 1)], crs="EPSG:28992")
    result = check_crs(gdf)
    assert result.crs.to_epsg() == 28992
    assert result.geometry.equals(gdf.geometry)

def test_if_no_crs_defined():
    #if no crs defined, raise ValueError
    gdf = gpd.GeoDataFrame(geometry=[Point(1, 1)])
    
    with pytest.raises(ValueError, match="NO CRS defined"):
        check_crs(gdf)

def test_polygon_crs():
    poly = Polygon([(4.9, 52.9), (5.0, 52.9), (5.0, 53.0), (4.9, 53.0), (4.9, 52.9)])
    gdf = gpd.GeoDataFrame(geometry=[poly], crs="EPSG:4326")
    result = check_crs(gdf)
    assert result.crs.to_epsg() == 28992

def test_original_crs_preserved():
    #This test ensures that the original crs is not changed outside the function check_crs
    gdf = gpd.GeoDataFrame(geometry=[Point(1, 1)], crs="EPSG:4326")
    original_crs = gdf.crs
    results=check_crs(gdf)
    assert results.crs.to_epsg() == 28992
    assert gdf.crs == original_crs

def test_loaded_layers_crs():
    #check if loaded layers have no crs or wrong crs, and convert them to 28992
    AHN2_points, _ = load_layers(elevation_AHN2_path,layer='AHN2_Groningen_Points')

    assert AHN2_points.crs is not None
    assert AHN2_points.crs.to_epsg() == 28992


#####################################
#Tests for elevation change analysis#
#####################################
def test_match_elev_points_does_not_change_inputs():
    #this test make sure that the function match_by_xy_and_diff 
    #does not modidy the input data
    gdf_2010 = gpd.GeoDataFrame(
        {'z':[10,20]},
        geometry=[Point(1,1), Point(2,2)],
        crs="EPSG:28992"
    )

    gdf_2020 = gpd.GeoDataFrame(
        {'z':[5,8]},
        geometry=[Point(1,1), Point(2,2)],
        crs="EPSG:28992"
    )

    #create copies so original data can be compared after 
    gdf_2010_copy = gdf_2010.copy(deep=True)
    gdf_2020_copy = gdf_2020.copy(deep=True)

    result = match_by_xy_and_diff(gdf_2010, gdf_2020,value_column='z')
    assert gdf_2010.equals(gdf_2010_copy)
    assert gdf_2020.equals(gdf_2020_copy)

def test_match_points_drop_nullgeometry_and_nanvalues():
    gdf_2010 = gpd.GeoDataFrame(
        {'z':[10,np.nan,30]},
        geometry=[Point(1,1), Point(2,2), None],
        crs="EPSG:28992"
    )

    gdf_2020 = gpd.GeoDataFrame(
        {'z':[5,None,15]},
        geometry=[Point(1,1), Point(2,2), Point(3,3)],
        crs="EPSG:28992"
    )

    result = match_by_xy_and_diff(gdf_2010, gdf_2020, value_column='z', as_geodataframe=False)
    assert len(result) == 1
    assert float(result['elev_AHN2'].iloc[0]) == 10
    assert float(result['elev_AHN4'].iloc[0]) == 5
    assert float(result['elev_diff'].iloc[0]) == 5


def test_math_inner_join():
    gdf_2010 = gpd.GeoDataFrame(
        {'z':[10,20]},
        geometry=[Point(1,1), Point(2,2)],
        crs="EPSG:28992"
    )

    gdf_2020 = gpd.GeoDataFrame(
        {'z':[5]},
        geometry=[Point(1,1)],
        crs="EPSG:28992"
    )

    result = match_by_xy_and_diff(gdf_2010, gdf_2020, value_column='z', as_geodataframe=False)


    assert len(result) == 1 #only one matching point
    assert set(result['x']) == {1}
    assert set(result['y']) == {1}




###############################
#Tests for population analysis#
###############################

def test_clean_population_layers():
    path_2010 = "F:/master/mod_2/Sci_Prog_For_Geospatial_Sciences/Pair assignment/data/Pop_2010.gpkg"

    input_gdf=gpd.read_file(path_2010,layer='Pop_2010')
    cleaned_gdf = clean_population_layers(input_gdf, 2010)

    assert isinstance(cleaned_gdf,gpd.GeoDataFrame)
    assert 'grid_id' in cleaned_gdf.columns
    assert 'pop_count' in cleaned_gdf.columns
    
    #ensure no data values are removed
    assert not cleaned_gdf['pop_count'].isin([-99997,-99998]).any()


#############################################
#Test function request boundary layer########
#############################################
def test_get_feature():
    endpoint = 'https://service.pdok.nl/kadaster/bestuurlijkegebieden/wfs/v1_0?'
    response=get_feature(endpoint)
        
    assert isinstance(response, gpd.GeoDataFrame)
        
    assert not response.empty

#############################################
#Test Generating polygons top 5 clusters#####
#############################################

def test_polygon_clusters():
    gdf = gpd.GeoDataFrame(geometry=[Point(1 + i * 10, 1) for i in range(5)],crs="EPSG:28992")
    gdf['cluster']=1
    polygons = polygon_clusters(gdf)
    
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert isinstance(polygons, gpd.GeoDataFrame)
    assert gdf.crs == 28992
    assert polygons.crs == 28992
    
    assert not gdf.empty
    assert not polygons.empty

#############################################
#Test CLipping datasets to the 5 polygons####
#############################################
    def test_clip_pol():
        polygon = Polygon([(5, 0), (10, 0), (10, 5), (0, 5)])
        gdf_poly = gpd.GeoDataFrame(geometry=[poly], crs="EPSG:28992")
        clip_poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        gdf_clip = gpd.GeoDataFrame(geometry=[clip_poly],crs="EPSG:28992")
        
        clip_geom = clip_pol(gdf_poly,gdf_clip)
        
        assert isinstance(gdf_poly, gpd.GeoDataFrame)
        assert isinstance(gdf_clip, gpd.GeoDataFrame)
        assert isinstance(clip_geom, gpd.GeoDataFrame)
        assert gdf_poly.geometry.is_valid.all()
        assert gdf_clip.geometry.is_valid.all()
        assert clip_geom.geometry.is_valid.all()
        
        assert not gdf_poly.empty
        assert not gdf_clip.empty
        assert not clip_geom.empty
        assert not gdf_poly.geometry.isnull().any()
        assert not gdf_clip.geometry.isnull().any()

#############################################
#Test Filtering the Province boundary#
#############################################

def test_filter_province():
    endpoint = 'https://service.pdok.nl/kadaster/bestuurlijkegebieden/wfs/v1_0?'
    response=get_feature(endpoint)
    province = 'groningen'
    filter=filter_province(response, province)
    
    assert isinstance(response, gpd.GeoDataFrame)
    assert isinstance(filter, gpd.GeoDataFrame)
    assert (filter["ligtInProvincieNaam"].str.lower() == province).any()
    
    assert not response.empty
    assert not filter.empty
    assert not (filter["ligtInProvincieNaam"].str.lower() != province).any()

#############################################
#Test merge the polygons for Groningen#######
#############################################
def test_merg_province():
    endpoint = 'https://service.pdok.nl/kadaster/bestuurlijkegebieden/wfs/v1_0?'
    response=get_feature(endpoint)
    province = 'groningen'
    filter=filter_province(response, province)
    result = merg_province(filter)
    
    assert isinstance(result, gpd.GeoDataFrame)
    assert result.geometry.is_valid.all()
        
    assert not result.empty
    assert not result.geometry.isnull().any()

#############################################
#Test Area Ha#
#############################################

def test_area_ha():
    pol = Polygon([(5, 0), (35, 0), (35, 5), (5, 5), (5, 0)])
    gdf = gpd.GeoDataFrame(geometry=[pol],crs="EPSG:28992")
    area = area_ha(gdf)
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert gdf.geometry.is_valid.all()
    assert area['area_ha'].iloc[0] == 0.015
        
    assert not gdf.empty
    assert not gdf.geometry.isnull().any()
    assert not area['area_ha'].iloc[0] != 0.015

######################################################
#Test Function  if a column exist in a Geodataframe#
######################################################
def test_create_field_pol():
    pol = Polygon([(5, 0), (35, 0), (35, 5), (5, 5), (5, 0)])
    gdf = gpd.GeoDataFrame(geometry=[pol],crs="EPSG:28992")
    column_name = 'NBBG20'
    result = create_field_pol(gdf, column_name)
    
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert isinstance(result, gpd.GeoDataFrame)
    assert column_name not in gdf.columns, f"Column '{column_name}' is missing in gdf"

    assert not gdf.empty
    assert not gdf.geometry.isnull().any()

##########################################
#Test plotting classes as a map###########
##########################################
def test_plot_classes():
    pol = Polygon([(5, 0), (35, 0), (35, 5), (5, 5), (5, 0)])
    gdf = gpd.GeoDataFrame(geometry=[pol],crs="EPSG:28992")
    gdf['categorie'] = ['Grass']
    gdf = area_ha(gdf)
    categorie = 'categorie'
    titles = 'test title'
    plot_classes(gdf, categorie, titles)
    
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert isinstance(categorie, str)
    assert isinstance(titles, str)
    
    assert not gdf.empty

######################################
#Test plotting classes as a bar chart#
######################################
    
def test_plot_bar_classes():
    pol = Polygon([(5, 0), (35, 0), (35, 5), (5, 5), (5, 0)])
    gdf = gpd.GeoDataFrame(geometry=[pol],crs="EPSG:28992")
    gdf['categorie'] = ['Grass']
    gdf = area_ha(gdf)
    categorie = 'categorie'
    titles = 'test title'
    plot_bar_classes(gdf, categorie, titles)
    
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert isinstance(categorie, str)
    assert isinstance(titles, str)
    
    assert not gdf.empty

#####################################################################################
#Test function  if a column exist in a Geodataframe and update the value for points#
##################################################################################### 
def test_update_field():
    points = [Point(5, 0), Point(35, 0), Point(35, 5), Point(5, 5)]
    gdf = gpd.GeoDataFrame(geometry=points, crs="EPSG:28992")
    gdf['categorie'] = ['Grass', 'Grass', 'Grass', 'Grass']
    column_name = 'categorie'
    original_name = 'Grass'
    final_name = 'Trees'
    result = update_field(gdf, column_name, original_name, final_name)
    
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert isinstance(column_name, str)
    assert isinstance(original_name, str)
    assert isinstance(final_name, str)
    assert isinstance(result, gpd.GeoDataFrame)
    assert column_name in result.columns and final_name in result[column_name].values, f"Column '{column_name}' is missing or value '{final_name}' not found"
    
    assert not gdf.empty
    assert not result.empty

#########################################
#Test function create a field for points#
#########################################
def test_create_field_pt():
    
    points = [Point(5, 0), Point(35, 0), Point(35, 5), Point(5, 5)]
    gdf = gpd.GeoDataFrame(geometry=points, crs="EPSG:28992")
    gdf['brogufvolledigeset — lifespan_start_time'] = ['2015-05-24', '2015-05-24', '2015-05-24', '2015-05-24']
    name_col = 'date'
    result = create_field_pt(gdf, name_col)
    
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert isinstance(result, gpd.GeoDataFrame)
    assert isinstance(name_col, str)
    assert name_col in result.columns, f"Column '{name_col}' is missing in gdf"
    assert '2011-2022' in result[name_col].values, f"Value '2011-2022' not found"
    
    assert not gdf.empty
    assert not gdf.geometry.isnull().any()

#########################################
#Test plotting classes as point as a map#
#########################################

def test_plot_classes_point():
    points = [Point(5, 0), Point(35, 0), Point(35, 5), Point(5, 5)]
    gdf = gpd.GeoDataFrame(geometry=points, crs="EPSG:28992")
    gdf['categorie'] = ['Grass', 'Grass', 'Grass', 'Grass']
    pol = Polygon([(5, 0), (35, 0), (35, 5), (5, 5), (5, 0)])
    gdf1 = gpd.GeoDataFrame(geometry=[pol],crs="EPSG:28992")
    categorie = 'categorie'
    base_layer = None
    overlay_layers = gdf1
    titles = 'test title'
    plot_classes_point(gdf, categorie, titles, base_layer=None, overlay_layers=None)
       
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert isinstance(gdf1, gpd.GeoDataFrame)
    assert isinstance(categorie, str)
    assert isinstance(titles, str)
    
    assert not gdf.empty
    assert not gdf1.empty
    
###############################################
#Test plotting classes as point as a bar chart#
###############################################

def test_plot_bar_classes_point():
    points = [Point(5, 0), Point(35, 0), Point(35, 5), Point(5, 5)]
    gdf = gpd.GeoDataFrame(geometry=points, crs="EPSG:28992")
    gdf['brogufvolledigeset — lifespan_start_time'] = ['2015-05-24', '2015-05-24', '2015-05-24', '2015-05-24']
    name_col = 'date_interval'
    result = create_field_pt(gdf, name_col)
    plot_bar_classes_point(result)
    
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert isinstance(result, gpd.GeoDataFrame)
    
    assert not gdf.empty
    assert not result.empty