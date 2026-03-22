import argparse
import urllib.request
import os
import tarfile
import shutil
import gzip


def download_if_missing(url, destination):
    if os.path.exists(destination):
        print(f"Skipping existing download: {destination}")
        return
    print(f"Downloading {url} -> {destination}")
    urllib.request.urlretrieve(url, destination)


def extract_member_if_missing(tar_path, member_name, out_path):
    if os.path.exists(out_path):
        print(f"Skipping existing extract target: {out_path}")
        return
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extract(member_name)
    os.rename(member_name, os.path.basename(out_path))
    shutil.move(os.path.basename(out_path), out_path)


# Parse command-line arguments
parser = argparse.ArgumentParser(description='Download and extract UniProt files')
parser.add_argument('year1', type=str, help='Year for testing Swissprot')
parser.add_argument('month1', type=str, help='Month for testing Swissprot')
parser.add_argument('year2', type=str, help='Year for pretraining and training')
parser.add_argument('month2', type=str, help='Month for pretraining and training')
parser.add_argument('year3', type=str, help='Year for ensemble training', default=None)
parser.add_argument('month3', type=str, help='Month for ensemble training', default=None)
args = parser.parse_args()

year1 = args.year1
month1 = args.month1
year2 = args.year2
month2 = args.month2
if args.year3:
    year3 = args.year3
    month3 = args.month3

# Set output directory
output_dir = "data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Download testing Swissprot
testing_url = f"https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/release-{year1}_{month1}/knowledgebase/uniprot_sprot-only{year1}_{month1}.tar.gz"
testing_tar = output_dir + f"/uniprot_sprot-only{year1}_{month1}.tar.gz"
download_if_missing(testing_url, testing_tar)

# Download pretraining and training
pretraining_url = f"https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/release-{year2}_{month2}/knowledgebase/knowledgebase{year2}_{month2}.tar.gz"
pretraining_tar = output_dir + f"/knowledgebase{year2}_{month2}.tar.gz"
download_if_missing(pretraining_url, pretraining_tar)

# Extract uniprot_sprot
extract_member_if_missing(
    testing_tar,
    "uniprot_sprot.dat.gz",
    output_dir + f"/uniprot_sprot{year1}_{month1}.data.gz",
)

# Extract knowledgebase
extract_member_if_missing(
    pretraining_tar,
    "uniprot_sprot.dat.gz",
    output_dir + f"/uniprot_sprot{year2}_{month2}.data.gz",
)
extract_member_if_missing(
    pretraining_tar,
    "uniprot_trembl.dat.gz",
    output_dir + f"/uniprot_trembl{year2}_{month2}.data.gz",
)

if year3:
    # Download ensemble Swissprot
    ensemble_url = f"https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/release-{year3}_{month3}/knowledgebase/uniprot_sprot-only{year3}_{month3}.tar.gz"
    ensemble_tar = output_dir + f"/uniprot_sprot-only{year3}_{month3}.tar.gz"
    download_if_missing(ensemble_url, ensemble_tar)
    # Extract uniprot_sprot for ensemble
    extract_member_if_missing(
        ensemble_tar,
        "uniprot_sprot.dat.gz",
        output_dir + f"/uniprot_sprot{year3}_{month3}.data.gz",
    )

# Download alphafold structures (PDB files) for SwissprotKB: 
pdb_url = f"https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/swissprot_pdb_v6.tar"
pdb_tar = output_dir + f"/swissprot_pdb_v6.tar"
download_if_missing(pdb_url, pdb_tar)

# download GO terms from: https://ftp.ebi.ac.uk/pub/databases/GO/goa/UNIPROT/goa_uniprot_all.gaf.gz and extract it
url = 'https://ftp.ebi.ac.uk/pub/databases/GO/goa/UNIPROT/goa_uniprot_all.gaf.gz'

goa_gz = output_dir + '/goa_uniprot_all.gaf.gz'
goa_txt = output_dir + '/goa_uniprot_all.gaf'
download_if_missing(url, goa_gz)
if not os.path.exists(goa_txt):
    with gzip.open(goa_gz, 'rb') as f_in:
        with open(goa_txt, 'wb') as f_out:
            f_out.write(f_in.read())
else:
    print(f"Skipping existing extract target: {goa_txt}")

