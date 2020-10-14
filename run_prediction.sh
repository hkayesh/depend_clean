#!/bin/bash


DATA_FILE=$1
DATASET=$2 # SITE-A / SITE-B

if [ -z "$1" ]; then
    echo "Please pass a csv data file"
    exit
fi

if [ "$DATASET" = "SITE-A" ]; then
    TRAINING_SYSTEM_A="site_a_dataset.csv"
    TRAINING_SYSTEM_B="r_site_a_dataset.csv"
    DATABASE_TYPE="site-a"
else 
    if [ "$DATASET" = "SITE-B" ]; then
        TRAINING_SYSTEM_A="site_b_dataset.csv"
        TRAINING_SYSTEM_B="r_site_b_dataset.csv"
        DATABASE_TYPE="site-b"
    else
        echo "Error: Invalid dataset type."
        exit
    fi
fi

cd system_a/ 
python analyzer_main.py --train files/$TRAINING_SYSTEM_A --data ../$DATA_FILE --output ../system_c/files/output_system_a.csv 
cd ../system_b/ 
Rscript main.R --train files/$TRAINING_SYSTEM_B --data ../$DATA_FILE --dataset $DATABASE_TYPE --output ../system_c/files/output_system_b.csv  
cd ../system_c/ 
python main.py --a files/output_system_a.csv --b files/output_system_b.csv --dataset $DATABASE_TYPE --output ../output.csv
