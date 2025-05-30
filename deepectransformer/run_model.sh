conda env create -f deepectransformer/environment.yml
conda activate deepectransformer

cd deepectransformer
mkdir -p ../results/deepectransformer
python3 run_deepectransformer.py -i ../data/price-149.fasta -o ../results/deepectransformer -g cpu -b 128 -cpu 112
# rename tmp folder to price-149
mv ../results/deepectransformer/tmp ../results/deepectransformer/price-149
# remove tmp folder
rm -rf ../results/deepectransformer/tmp
python3 run_deepectransformer.py -i ../data/test_ec.fasta -o ../results/deepectransformer -g cpu -b 128 -cpu 112
# rename tmp folder to test
mv ../results/deepectransformer/tmp ../results/deepectransformer/test
# remove tmp folder
rm -rf ../results/deepectransformer/tmp
python3 run_deepectransformer.py -i ../data/ens-30.fasta -o ../results/deepectransformer -g cpu -b 128 -cpu 112
# rename tmp folder to ens-30
mv ../results/deepectransformer/tmp ../results/deepectransformer/ens-30
# remove tmp folder
rm -rf ../results/deepectransformer/tmp 
python3 run_deepectransformer.py -i ../data/ens-100.fasta -o ../results/deepectransformer -g cpu -b 128 -cpu 112
# rename tmp folder to ens-100
mv ../results/deepectransformer/tmp ../results/deepectransformer/ens-100
# remove tmp folder
rm -rf ../results/deepectransformer/tmp

deactivate deepectransformer
