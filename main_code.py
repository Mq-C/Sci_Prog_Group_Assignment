from src.functions import (match_by_xy_and_diff,
                            load_layers, 
                            check_crs,
                            KD_clustering,
                            plot_clusters,
                            between_20_and_50cm_diff,
                            population_analysis,
                            polygon_clusters)
import matplotlib.pyplot as plt


elevation_AHN2_path = "F:/master/mod_2/Sci_Prog_For_Geospatial_Sciences/Pair assignment/data/AHN2_Groningen_Points.gpkg"
elevation_AHN4_path = "F:/master/mod_2/Sci_Prog_For_Geospatial_Sciences/Pair assignment/data/AHN4_Groningen_Points.gpkg"
population_2010_path = "F:/master/mod_2/Sci_Prog_For_Geospatial_Sciences/Pair assignment/data/Pop_2010.gpkg"
population_2020_path = "F:/master/mod_2/Sci_Prog_For_Geospatial_Sciences/Pair assignment/data/Pop_2020.gpkg"
groningen_boundary_path = 'F:/master/mod_2/Sci_Prog_For_Geospatial_Sciences/Pair assignment/data/Groningen_boundary/Groningen_28992.gpkg'

data_paths = {
    'ahn2': elevation_AHN2_path ,
    'ahn4': elevation_AHN4_path,
    'pop_2010':population_2010_path,
    'pop_2020':population_2020_path,
    'groningen_boundary':groningen_boundary_path
}
run_elevation =True
run_population=True


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


def main():
    layers ={
        'ahn2':load_layers(elevation_AHN2_path,layer ='AHN2_Groningen_Points')[0],
        'ahn4':load_layers(elevation_AHN4_path,layer ='AHN4_Groningen_Points')[0],
        'pop_2010':load_layers(population_2010_path,layer='Pop_2010')[0],
        'pop_2020':load_layers(population_2020_path,layer='Pop_2020')[0],
        'groningen_boundary':load_layers(groningen_boundary_path,layer='Groningen_28992')[0]

    }



    for name,gdf in layers.items():
        check_crs(gdf)

    generated_polygons=None

    if run_elevation:
        cluster_data=run_elevation_analysis(layers)
        plot_clusters(cluster_data,layers['groningen_boundary'])

        print('---Generating polygons for top 5 clusters---')
        generated_polygons = polygon_clusters(cluster_data)
    
    if run_population:
        if generated_polygons is not None:
            print('---Performing population analysis---')
            clipped_pop_gdf = run_population_analysis(layers,generated_polygons)

if __name__ == "__main__":
    main()