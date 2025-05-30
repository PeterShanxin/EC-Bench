# Download and extract CatFam data
echo "Downloading CatFam..."
wget -O catfam.tar.gz http://www.bhsai.org/downloads/catfam.tar.gz

echo "Extracting CatFam into Catfam/ folder..."
tar -xzf catfam.tar.gz -C CatFam

# Clean up
rm catfam.tar.gz
echo "CatFam setup completed."

cd CatFam
source/catsearch.pl -d CatFamDB/CatFam_v2.0/CatFam4D99R -i ../data/price-149.fasta -o results/price-149.output
source/catsearch.pl -d CatFamDB/CatFam_v2.0/CatFam4D99R -i ../data/test_ec.fasta -o results/test.output
echo "CatFam search completed. Results are in results/ directory."
# Move results to a specific directory
mkdir -p ../results/catfam
mv results/ ../results/catfam/

cd ../
