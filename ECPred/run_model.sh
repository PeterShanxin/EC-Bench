# Update package lists
sudo apt-get update

# Install Java Runtime Environment
sudo apt-get install -y default-jre

# Install Java Development Kit
sudo apt-get install -y default-jdk
sudo apt-get install build-essential

# Download the file from the shortened URL
wget https://goo.gl/g2tMJ4

# Unzip contents into the ECPred folder
tar -xvf ECPred.tar.gz -C ECPred

echo "Download and extraction to ECPred complete."

cd ECPred
./runLinux.sh 
mkdir -p ../results/ECPred
java -jar ECPred.jar weighted ../data/price-149.fasta ../ECPred/ temp/ results_price_149.tsv
java -jar ECPred.jar weighted ../data/test_ec.fasta ../ECPred/ temp/ results_ec.tsv
mv results_price_149.tsv ../results/ECPred/
mv results_ec.tsv ../results/ECPred/
echo "ECPred search completed. Results are in ../results/ECPred"
cd ../