from src.functions import (match_by_xy_and_diff,
                            load_layers, 
                            check_crs,
                            KD_clustering,
                            plot_clusters,
                            between_20_and_50cm_diff,
                            population_analysis,
                            polygon_clusters,
                            get_feature,
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
                            )
import matplotlib.pyplot as plt
import pandas as pd

#elevation_AHN2_path = "F:/master/mod_2/Sci_Prog_For_Geospatial_Sciences/Pair assignment/data/AHN2_Groningen_Points.gpkg"
#elevation_AHN4_path = "F:/master/mod_2/Sci_Prog_For_Geospatial_Sciences/Pair assignment/data/AHN4_Groningen_Points.gpkg"
#population_2010_path = "F:/master/mod_2/Sci_Prog_For_Geospatial_Sciences/Pair assignment/data/Pop_2010.gpkg"
#population_2020_path = "F:/master/mod_2/Sci_Prog_For_Geospatial_Sciences/Pair assignment/data/Pop_2020.gpkg"
#groningen_boundary_path = 'F:/master/mod_2/Sci_Prog_For_Geospatial_Sciences/Pair assignment/data/Groningen_boundary/Groningen_28992.gpkg'
elevation_AHN2_path = r"C:/MGEO/Y1/Q2/Scientific Prog Geospatial Sciences/Programming exercises/Week_6/GIT HUB exercise/Base_file/AHN2_Groningen_Points.gpkg"
elevation_AHN4_path = r"C:/MGEO/Y1/Q2/Scientific Prog Geospatial Sciences/Programming exercises/Week_6/GIT HUB exercise/Base_file/AHN4_Groningen_Points.gpkg"
population_2010_path = r"C:/MGEO/Y1/Q2/Scientific Prog Geospatial Sciences/Programming exercises/Week_6/GIT HUB exercise/Base_file/Pop_2010.gpkg"
population_2020_path = r"C:/MGEO/Y1/Q2/Scientific Prog Geospatial Sciences/Programming exercises/Week_6/GIT HUB exercise/Base_file/Pop_2020.gpkg"
endpoint = 'https://service.pdok.nl/kadaster/bestuurlijkegebieden/wfs/v1_0?'
Land_use_2010_path = r"C:\MGEO\Y1\Q2\Scientific Prog Geospatial Sciences\Programming exercises\Week_6\bestand_bodemgebruik_2010.gpkg"
Land_use_2020_path = r"C:\MGEO\Y1\Q2\Scientific Prog Geospatial Sciences\Programming exercises\Week_6\Land_use_2020.gpkg"
wells_path = r"C:\MGEO\Y1\Q2\Scientific Prog Geospatial Sciences\Programming exercises\Week_6\Groundwater_usage_facilty.gpkg"

data_paths = {
    'ahn2': elevation_AHN2_path ,
    'ahn4': elevation_AHN4_path,
    'pop_2010':population_2010_path,
    'pop_2020':population_2020_path,
    #'groningen_boundary':groningen_boundary_path,
    'land_2010':Land_use_2010_path,
    'land_2020':Land_use_2020_path,
    'wells': wells_path
}

# Functions to run (choose true or false)
run_elevation =True
run_population=True
run_boundaries=True
run_province=True
run_land_use_2010_province = True
run_land_use_2020_province = True
run_land_use_2010_clusters = True
run_land_use_2020_clusters = True
run_Groningen_wells = True
run_clusters_wells = True

def run_elevation_analysis(layers):
    print("---Performing elevation difference analysis between AHN2 and AHN4---")
    result = match_by_xy_and_diff(layers['ahn2'], layers['ahn4'], value_column='VALUE', as_geodataframe=True)

    sinking_points_df = between_20_and_50cm_diff(result)
    print(f"Number of sinking points (elev_diff between 20 and 50cm): {len(sinking_points_df)}")

    #KD tree
    clustered_data = KD_clustering(sinking_points_df)
    return clustered_data
   

def run_population_analysis(layers,generated_polygons):
    print("---Matching population polygons and clipping it to cluster polygons---")
    clipped_pop_gdf,trend_summary = population_analysis(
        layers['pop_2010'],
        layers['pop_2020'],
        generated_polygons)
    
    #plotting population polygons within the 5 clusters
    fig,ax =plt.subplots(figsize=(10,10))
    generated_polygons.plot(ax=ax,facecolor='none',edgecolor='black',linewidth=1)
    clipped_pop_gdf.plot(ax=ax,column='pop_diff',
                         cmap='RdYlGn',
                         legend=True,
                         edgecolor='none',
                         legend_kwds={'label':'Change in number of residents (2010-2020)','orientation':'horizontal'})
    
    plt.title('Population Dynamics within High-Subsidence Zones (Groningen)')
    plt.xlabel('RD Easting (m)', fontsize=10)
    plt.ylabel('RD Northing (m)', fontsize=10)
    plt.show()
    return clipped_pop_gdf

def boundaries(endpoint):
    print(f'---Getting features from {endpoint}---')
    gdf_adm = get_feature(endpoint)
    return gdf_adm

def Groningen_Province(gdf_adm):
    gdf_adm= check_crs(gdf_adm)
    gdf_adm_fil = filter_province(gdf_adm, 'groningen')
    gdf_adm_gron=merg_province(gdf_adm_fil)
    print(f'records after merging: {len(gdf_adm_gron)}')
    return gdf_adm_gron
    
def land_use_2010_province(layers, gdf_adm_gron):
    gdf2010=clip_pol(layers['land_2010'],gdf_adm_gron)
    gdf2010=area_ha(gdf2010)
    print(f'---plotting in a map: Land use 2010 Groningen---')
    plot_classes(gdf2010, 'categorie', 'Land use 2010 Groningen')
    print(f'---plotting in a bar chart: Land use 2010 Groningen---')
    plot_bar_classes(gdf2010, 'categorie', 'Land use 2010 Groningen')

def land_use_2020_province(layers, gdf_adm_gron):
    gdf2020=create_field_pol(layers['land_2020'], 'NBBG20')
    gdf2020=clip_pol(gdf2020,gdf_adm_gron)
    gdf2020=area_ha(gdf2020)
    print(f'---plotting in a map: Land use 2020 Groningen---')
    plot_classes(gdf2020, 'categorie', 'Land use 2020 Groningen')
    print(f'---plotting in a bar chart: Land use 2020 Groningen---')
    plot_bar_classes(gdf2020, 'categorie', 'Land use 2020 Groningen')
    return gdf2020
    
def land_use_2010_clusters(layers, generated_polygons):
    gdf2010_clip = clip_pol(layers['land_2010'], generated_polygons)
    gdf2010_clip=area_ha(gdf2010_clip)
    print(f'---plotting in a map: Land use 2010 clusters in Groningen---')
    plot_classes(gdf2010_clip, 'categorie', 'Land use 2010 for area with the most sinking points')
    print(f'---plotting in a bar chart: Land use 2010 clusters in Groningen---')
    plot_bar_classes(gdf2010_clip, 'categorie', 'Land use 2010 for area with the most sinking points')
    
def land_use_2020_clusters(generated_polygons):
    gdf2020_clip = land_use_2020_province().copy
    gdf2020_clip = clip_pol(gdf2020_clip, generated_polygons)
    gdf2020_clip=area_ha(gdf2020_clip)
    print(f'---plotting in a map: Land use 2020 clusters in Groningen---')
    plot_classes(gdf2020_clip, 'categorie', 'Land use 2020 for area with the most sinking points')
    print(f'---plotting in a bar chart: Land use 2020 clusters in Groningen---')
    plot_bar_classes(gdf2020_clip, 'categorie', 'Land use 2020 for area with the most sinking points')

def wells_Groningen_analysis(layers, gdf_adm_gron):
    gdf_wll = layers['wells']
    gdf_wll=clip_pol(gdf_wll,gdf_adm_gron)
    gdf_wll=update_field(gdf_wll, 'delivery_context', 'wetMilieubeheerOfBesluitLozenBuitenInrichtingen', 'other')
    gdf_wll=create_field_pt(gdf_wll, 'date_interval')
    print(f'---plotting in a map: Wells in Groningen grouped by date---')
    plot_classes_point(gdf_wll, 'date_interval', 'Wells in Groningen grouped by Date',base_layer=gdf_adm_gron, overlay_layers=None)
    print(f'---plotting in a bar chart: Wells in Groningen grouped by Date---')
    plot_bar_classes_point(gdf_wll)
    return gdf_wll

def wells_clusters_analysis(generated_polygons, base_layer, overlay_layers):
    print('---Wells inside the clusters before 2010---')
    gdf_wll = wells_Groningen_analysis().copy
    gdf_wll_2010 = gdf_wll[gdf_wll['date_interval'] == '<=2010'].copy()
    gdf_wll_2010 = clip_pol(gdf_wll_2010, generated_polygons)
    print(f'---plotting in a map: Wells inside the clusters before 2010---')
    plot_classes_point(gdf_wll_2010, 'date_interval', 'Wells inside the polygons before 2010',base_layer=base_layer, overlay_layers=generated_polygons)
    print(f'---plotting in a bar chart: Wells inside the clusters before 2010---')
    plot_bar_classes_point(gdf_wll_2010)
    
    print('---Wells inside the clusters between 2010 - 2022---')
    gdf_wll = wells_Groningen_analysis().copy
    gdf_wll_2020 = gdf_wll[gdf_wll['date_interval'] == '2011-2022'].copy()
    gdf_wll_2020 = clip_pol(gdf_wll_2020, generated_polygons)
    print(f'---plotting in a map: Wells inside the clusters between 2010 - 2022---')
    plot_classes_point(gdf_wll_2020, 'date_interval', 'Wells inside the polygons before 2010',base_layer=base_layer, overlay_layers=generated_polygons)
    print(f'---plotting in a bar chart: Wells inside the clusters between 2010 - 2022---')
    plot_bar_classes_point(gdf_wll_2020)
    
    
def main():
    layers ={
        'ahn2':load_layers(elevation_AHN2_path,layer ='AHN2_Groningen_Points')[0],
        'ahn4':load_layers(elevation_AHN4_path,layer ='AHN4_Groningen_Points')[0],
        'pop_2010':load_layers(population_2010_path,layer='Pop_2010')[0],
        'pop_2020':load_layers(population_2020_path,layer='Pop_2020')[0],
        #'groningen_boundary':load_layers(groningen_boundary_path,layer='Groningen_28992')[0],
        'land_2010':load_layers(Land_use_2010_path, layer='bestand_bodemgebruik_2010')[0],
        'land_2020':load_layers(Land_use_2020_path, layer='Land_use_2020')[0],
        'wells': load_layers(wells_path, layer='Groundwater_usage_facility_2')[0]
    }



    for name,gdf in layers.items():
        check_crs(gdf)

    generated_polygons=None


    if run_boundaries:
        gdf_adm= boundaries(endpoint)
    
    if run_province:
        print('---Getting Province Boundary---')
        gdf_adm_gron = Groningen_Province(gdf_adm)
        
    if run_elevation:
        cluster_data=run_elevation_analysis(gdf_adm_gron)
        #cluster_data=run_elevation_analysis(layers)
        #plot_clusters(cluster_data,layers['groningen_boundary'])
        plot_clusters(cluster_data,gdf_adm_gron)
        print('---Generating polygons for top 5 clusters---')
        generated_polygons = polygon_clusters(cluster_data)
    
    if run_population:
        if generated_polygons is not None:
            print('---Performing population analysis---')
            clipped_pop_gdf = run_population_analysis(layers,generated_polygons)
       
    if run_land_use_2010_province:
        print('---Land use 2010 Groningen Province---')
        land_use_2010_province(layers, gdf_adm_gron)
        
    if run_land_use_2020_province:
        print('---Land use 2020 Groningen Province---')
        land_use_2020_province(layers, gdf_adm_gron)
    
    if run_land_use_2010_clusters:
        print('---Land use 2010 Groningen clusters in Groningen---')
        land_use_2010_clusters(layers, generated_polygons)
        
    if run_land_use_2020_clusters:
        print('---Land use 2020 Groningen clusters in Groningen---')
        land_use_2020_clusters(layers, generated_polygons)
        
    if run_Groningen_wells:
        print('---Wells in Groningen grouped by date---')
        wells_Groningen_analysis(layers, gdf_adm_gron)
    
    if run_clusters_wells:
        print('---Wells in clusters by date---')
        wells_clusters_analysis(layers, gdf_adm_gron, generated_polygons)
    
if __name__ == "__main__":
    main()