conda env create -f DeepEC/environment.yml
conda activate deepec
cd DeepEC

python3 deepec.py -i ../data/price-149.fasta -o ../results/DeepEC
mv ../results/DeepEC/DeepEC_Result.txt ../results/DeepEC/DeepEC_Result_price_149.txt

python3 deepec.py -i ../data/test_ec.fasta -o ../results/DeepEC
mv ../results/DeepEC/DeepEC_Result.txt ../results/DeepEC/DeepEC_Result_test.txt

cd ../
deactivate deepec
