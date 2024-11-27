#!/bin/bash

cd Few-Shot || { echo "Failed to cd into Few-Shot"; exit 1; }
bash inference.sh

cd .. || { echo "Failed to cd out of Few-Shot"; exit 1; }
cd SFT || { echo "Failed to cd into SFT"; exit 1; }
bash inference.sh || { echo "Failed to run combine_all.sh in SFT"; exit 1; }

cd .. || { echo "Failed to cd out of SFT"; exit 1; }
cd DPO || { echo "Failed to cd into DPO"; exit 1; }
bash inference.sh || { echo "Failed to run combine_all.sh in DPO"; exit 1; }