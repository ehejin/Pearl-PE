## Datasets

Download the DrugOOD datasets [here](https://drive.google.com/drive/folders/17nVALCgTz0LV8pVuoM0xQnRqwRH3Bz7a?usp=drive_link). 

## Running experiments

To run experiments on ZINC, cd to ./zinc and run
```
python runner.py --config_dirpath ../configs/zinc --config_name BPEARL_ZINC.yaml --seed 0
python runner.py --config_dirpath ../configs/zinc --config_name RPEARL-ZINC.yaml --seed 0
```

To run experiments on DrugOOD, cd to ./drugood and run
```
python runner.py --config_dirpath ../configs/drugood/assay --config_name PEARL_gine_gin.yaml --dataset assay --seed 0
python runner.py --config_dirpath ../configs/drugood/scaffold --config_name PEARL_GINE_GIN.yaml --dataset scaffold --seed 0
python runner.py --config_dirpath ../configs/drugood/scaffold --config_name PEARL_GINE_GIN.yaml --dataset size --seed 0 
```

To run experiments on the Peptides-struct dataset, cd to ./peptides and run
```
python runner.py --config_dirpath ../configs/zinc --config_name BPEARL-peptides.yaml 
python runner.py --config_dirpath ../configs/zinc --config_name RPEARL-peptides.yaml 
```

## Attribution
This code framework is from the SPE repository by [Huang et al., 2024](https://github.com/Graph-COM/SPE).