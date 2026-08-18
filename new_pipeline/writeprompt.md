python3 new_pipeline/get_new_data.py --shapefile "/Users/braydenwinnicki/Downloads/cb_2023_25_tract_500k/cb_2023_25_tract_500k.shp"  --fips 25 --mode multi --census-api-link "https://api.census.gov/data/2023/acs/acs5"

python3 new_pipeline/run_experiment.py  --cache  new_pipeline/data/cache/09_tracts_multi.pt --csv  new_pipeline/data/data_csvs/processed_09_tracts_multi.csv --model resnet_l4 --mode eval --epochs 25 --batch-size 8 --lr 0.001 --random-state 42

python3 new_pipeline/run_experiment.py --cache new_pipeline/data/cache/25_tracts_multi.pt --csv new_pipeline/data/data_csvs/25_tracts_multi.csv --model resnet_l4 --mode eval --weights new_pipeline/data/models/09_resnet_l4_multi.pth --batch-size 8