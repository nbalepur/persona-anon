#!/bin/bash

ALL_USE_PERSONA=('True' 'False')
ALL_RESPONSE_TYPE=('chosen')

for use_persona in "${ALL_USE_PERSONA[@]}"; do
    for response_type in "${ALL_RESPONSE_TYPE[@]}"; do
        echo "Running with use_persona=$use_persona and response_type=$response_type"
        python inference.py --use_persona "$use_persona" --response_type "$response_type"
    done
done

# ALL_USE_PERSONA=('True' 'False')
# ALL_RESPONSE_TYPE=('chosen')

# for use_persona in "${ALL_USE_PERSONA[@]}"; do
#     for response_type in "${ALL_RESPONSE_TYPE[@]}"; do
#         echo "Running with use_persona=$use_persona and response_type=$response_type"
#         python inference.py --use_persona "$use_persona" --response_type "$response_type"
#     done
# done
