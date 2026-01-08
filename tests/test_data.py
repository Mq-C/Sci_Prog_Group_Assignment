import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon
from src.functions import (load_layers, 
                           check_crs,
                           match_by_xy_and_diff,
                           clean_population_layers)
import numpy as np




elevation_AHN2_path = "F:/master/mod_2/Sci_Prog_For_Geospatial_Sciences/Pair assignment/data/AHN2_Groningen_Points.gpkg"
elevation_AHN4_path = "F:/master/mod_2/Sci_Prog_For_Geospatial_Sciences/Pair assignment/data/AHN4_Groningen_Points.gpkg"
population_2010_path = "F:/master/mod_2/Sci_Prog_For_Geospatial_Sciences/Pair assignment/data/Pop_2010.gpkg"
population_2020_path = "F:/master/mod_2/Sci_Prog_For_Geospatial_Sciences/Pair assignment/data/Pop_2020.gpkg"
#############################################
#Test for loading and checking CRS of layers#
#############################################
def test_layers_load():

    AHN2_points = load_layers(elevation_AHN2_path)
    AHN4_points = load_layers(elevation_AHN4_path)
    pop_2010  = load_layers(population_2010_path)
    pop_2020  = load_layers(population_2020_path)

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
    AHN2_points = load_layers(elevation_AHN2_path)

    assert AHN2_points.crs is not None
    assert AHN2_points.crs.to_epsg() == 28992

def test_loaded_poly_is_valid():
    pop_2010  = load_layers(population_2010_path)
    pop_2020  = load_layers(population_2020_path)
    assert pop_2010.is_valid.all()
    assert pop_2020.is_valid.all()

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
    path_2010 = '"F:/master/mod_2/Sci_Prog_For_Geospatial_Sciences/Pair assignment/data/Pop_2010.gpkg"'

    pop_2010 = clean_population_layers(population_2010_path,2010)

    assert isinstance(pop_2010,gpd.GeoDataFrame)
    assert 'grid_id' in pop_2010.columns
    assert 'pop_count' in pop_2010.columns
    
    #ensure no data values are removed
    assert not pop_2010['pop_count'].isin([-99997,-99998]).any()
