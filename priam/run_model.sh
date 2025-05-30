#!/bin/bash

# Set installation directory
BLAST_DIR=blast-2.2.26
mkdir -p $BLAST_DIR

# Check if BLAST is already installed
if [ ! -f "$BLAST_DIR/blastall" ]; then
    echo "Installing BLAST 2.2.26..."

    cd /tmp
    wget ftp://ftp.ncbi.nlm.nih.gov/blast/executables/release/2.2.26/ncbi-blast-2.2.26-x64-linux.tar.gz
    tar -xzf ncbi-blast-2.2.26-x64-linux.tar.gz
    mv ncbi-blast-2.2.26/* $BLAST_DIR
    rm -rf ncbi-blast-2.2.26 ncbi-blast-2.2.26-x64-linux.tar.gz

    echo "BLAST 2.2.26 installed at $BLAST_DIR"
else
    echo "BLAST 2.2.26 is already installed at $BLAST_DIR"
fi

# Add to PATH (optional, if not already)
export PATH=$BLAST_DIR:$PATH

cd priam
java -jar PRIAM_search.jar -n "price-149" -i ../data/price-149.fasta -p PRIAM_MAR13 -o "../results/priam" --pt 0 --mp 60 --cc T --bd ../blast-2.2.26/bin --np 112
java -jar PRIAM_search.jar -n "test" -i ../data/test_ec.fasta -p PRIAM_MAR13 -o "../results/priam" --pt 0 --mp 60 --cc T --bd ../blast-2.2.26/bin --np 112
mv RESULTS ../results/priam
echo "PRIAM search completed. Results are in ../results/priam"
cd ../