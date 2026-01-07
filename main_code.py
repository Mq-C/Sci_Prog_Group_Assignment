from src.functions import match_by_xy_and_diff, load_layers, check_crs,KD_clustering,plot_clusters,between_20_and_50cm_diff
from pathlib import Path
elevation_AHN2_path = "F:/master/mod_2/Sci_Prog_For_Geospatial_Sciences/Pair assignment/data/AHN2_Groningen_Points.gpkg"
elevation_AHN4_path = "F:/master/mod_2/Sci_Prog_For_Geospatial_Sciences/Pair assignment/data/AHN4_Groningen_Points.gpkg"

def main():
    ahn2 = load_layers(elevation_AHN2_path)
    ahn4 = load_layers(elevation_AHN4_path)

    ahn2 = check_crs(ahn2)
    ahn4 = check_crs(ahn4)

    groningen_boundary_path = 'F:/master/mod_2/Sci_Prog_For_Geospatial_Sciences/Pair assignment/data/Groningen_boundary/Groningen_28992.gpkg'

    print("Performing elevation difference analysis between AHN2 and AHN4")
    result = match_by_xy_and_diff(ahn2, ahn4, value_column='VALUE', as_geodataframe=True)

    sinking_points_df = between_20_and_50cm_diff(result)

    print(f"Number of sinking points (elev_diff between 20 and 50cm): {len(sinking_points_df)}")

    clustered_data = KD_clustering(sinking_points_df)
    print("Generating plot...")
    plot_clusters(clustered_data,groningen_boundary_path)

if __name__ == "__main__":
    main()