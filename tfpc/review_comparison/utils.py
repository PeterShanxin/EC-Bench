def generate_fasta_file(df, fasta_path):
    with open(fasta_path, "w") as fasta_file:
        for _, row in df.iterrows():
            uniprot_id = row["id_uniprot"]
            sequence = row["sequence"]
            fasta_file.write(">" + uniprot_id + "\n")
            fasta_file.write(sequence + "\n")
