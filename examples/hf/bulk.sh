#! /bin/bash

# This script is used to run the bulk calculation
# four bulk files:
# bulk_test_CH4_1.py
# bulk_test_CH4_2.py
# bulk_test_HF_1.py
# bulk_test_HF_2.py
# bulk_test_H2O_1.py
# bulk_test_H2O_2.py

# The first two files are for CH4, the last two files are for HF

conda activate intern
echo "Running bulk_test_CH4_1.py"
python bulk_test_CH4_1.py > bulk_CH4_1.log
wait
echo "Running bulk_test_CH4_2.py"
python bulk_test_CH4_2.py > bulk_CH4_2.log
wait
echo "Running bulk_test_HF_1.py"
python bulk_test_HF_1.py > bulk_HF_1.log
wait
echo "Running bulk_test_HF_2.py"
python bulk_test_HF_2.py > bulk_HF_2.log
wait
echo "Running bulk_test_H2O_1.py"
python bulk_test_H2O_1.py > bulk_H2O_1.log
wait
echo "Running bulk_test_H2O_2.py"
python bulk_test_H2O_2.py > bulk_H2O_2.log


