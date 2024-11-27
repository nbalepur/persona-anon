#!/bin/bash

ALL_USE_PERSONA=('True')
ALL_RESPONSE_TYPE=('rejected' 'all' 'chosen')

for use_persona in "${ALL_USE_PERSONA[@]}"; do
    for response_type in "${ALL_RESPONSE_TYPE[@]}"; do
        echo "Running with use_persona=$use_persona and response_type=$response_type"
        python train.py --use_persona "$use_persona" --response_type "$response_type"
    done
done

ALL_USE_PERSONA=('False')
ALL_RESPONSE_TYPE=('chosen')

for use_persona in "${ALL_USE_PERSONA[@]}"; do
    for response_type in "${ALL_RESPONSE_TYPE[@]}"; do
        echo "Running with use_persona=$use_persona and response_type=$response_type"
        python train.py --use_persona "$use_persona" --response_type "$response_type"
    done
done
