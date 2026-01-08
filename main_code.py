from src.functions import match_by_xy_and_diff, load_layers, check_crs,KD_clustering,plot_clusters,between_20_and_50cm_diff
from pathlib import Path


elevation_AHN2_path = "F:/master/mod_2/Sci_Prog_For_Geospatial_Sciences/Pair assignment/data/AHN2_Groningen_Points.gpkg"
elevation_AHN4_path = "F:/master/mod_2/Sci_Prog_For_Geospatial_Sciences/Pair assignment/data/AHN4_Groningen_Points.gpkg"
population_2010_path = "F:/master/mod_2/Sci_Prog_For_Geospatial_Sciences/Pair assignment/data\Pop_2010.gpkg"
population_2020_path = "F:/master/mod_2/Sci_Prog_For_Geospatial_Sciences/Pair assignment/data/Pop_2010.gpkg"
cluster_poly_path ="F:/master/mod_2/Sci_Prog_For_Geospatial_Sciences/Pair assignment/data/outputs/Polygons_top_clusters_Filtered.gpkg"
groningen_boundary_path = 'F:/master/mod_2/Sci_Prog_For_Geospatial_Sciences/Pair assignment/data/Groningen_boundary/Groningen_28992.gpkg'

data_paths = {
    'ahn2': elevation_AHN2_path,
    'ahn4': elevation_AHN4_path,
    'pop_2010':population_2010_path,
    'pop_2020':population_2020_path,
    'groningen_boundary':groningen_boundary_path
}

def main():
    layers ={name: load_layers(path) for name,path in data_paths.items()}
    for name,gdf in layers.items():
        check_crs(gdf)


    print("Performing elevation difference analysis between AHN2 and AHN4")
    result = match_by_xy_and_diff(layers['ahn2'], layers['ahn4'], value_column='VALUE', as_geodataframe=True)

    sinking_points_df = between_20_and_50cm_diff(result)

    print(f"Number of sinking points (elev_diff between 20 and 50cm): {len(sinking_points_df)}")

    clustered_data = KD_clustering(sinking_points_df)
    print("Generating plot")
    plot_clusters(clustered_data,layers['groningen_boundary'])

if __name__ == "__main__":
    main()