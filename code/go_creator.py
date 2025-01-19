import pandas as pd
import ahocorasick
import pickle

def filter_columns(input_file = 'goa_uniprot_all.gaf', output_file = 'filtered_goa.csv'):

    column_names = [
        'DB', 'DB_Object_ID', 'DB_Object_Symbol', 'Qualifier', 'GO_ID', 
        'DB_Reference', 'Evidence_Code', 'With_From', 'Aspect', 
        'DB_Object_Name', 'DB_Object_Synonym', 'DB_Object_Type', 
        'Taxon_and_Interacting_Taxon', 'Date', 'Assigned_By', 
        'Annotation_Extension', 'Gene_Product_Form_ID'
    ]

    # Initialize output file
    with open(output_file, 'w') as f:
        pass

    # Step 1: Read the DataFrame in chunks
    chunksize = 1000000  # Adjust the chunk size as needed
    chunk_c=1
    for chunk in pd.read_csv(input_file, sep='\t', names=column_names, comment='!', dtype=str, chunksize=chunksize):
        
        # Step 2: Keep only the 'DB_Object_ID' and 'GO_ID' columns
        df_filtered = chunk[['DB_Object_ID', 'GO_ID']]
        
        # Step 3: Rename the columns to 'id' and 'go_terms'
        df_filtered.columns = ['id', 'go_terms']
        
        # Step 4: Append the filtered and renamed DataFrame chunk to the output file
        df_filtered.to_csv(output_file, mode='a', index=False, header=False)
        print(chunk_c)
        chunk_c += 1

    print(f"The filtered DataFrame has been saved to {output_file}")
    
    
def filter_ids(filter_ids_file = 'data/pretrain_ec_ids.tsv', input_file = 'data/filtered_goa.csv', output_file = 'data/pretrain_go.csv'):

    filter_ids_df = pd.read_csv(filter_ids_file, sep='\t', header=None, names=['id'])
    filter_ids = filter_ids_df['id'].tolist()
    print(len(filter_ids))

    with open(output_file, 'w') as f:
        pass

    # Build the Aho-Corasick automaton
    A = ahocorasick.Automaton()
    for i in filter_ids:
        A.add_word(i, i)
    A.make_automaton()
       
    # Step 1: Read the DataFrame in chunks
    chunksize = 1000000  # Adjust the chunk size as needed
    for chunk in pd.read_csv(input_file, sep=',', header=None, names=['id', 'go_terms'], chunksize=chunksize):
        def filter_terms(row):
            return A.exists(row['id'])

        df_filtered = chunk[chunk.apply(filter_terms, axis=1)]
               
        # Step 5: Append the filtered and renamed DataFrame chunk to the output file
        df_filtered.to_csv(output_file, mode='a', index=False)
                
    print(f"The filtered DataFrame has been saved to {output_file}")


def filter_frequent_pretrain(input_file = 'data/pretrain_go.csv', output_file = 'data/pretrain_go_frequent.csv', go_terms_dict_file = 'data/frequent_go_terms.pkl'):
    # Step 1: Read the DataFrame from the input file
    df = pd.read_csv(input_file, sep=',')
    df.drop_duplicates(subset=['id', 'go_terms'], inplace=True)
    df.to_csv(input_file, index=False)
    
    # Step 2: Calculate the frequency of each GO term
    go_term_counts = df['go_terms'].value_counts()

    # Step 3: Filter out rows with GO terms that appear less than 100 times
    frequent_go_terms = go_term_counts[go_term_counts >= 100].index

    # Build the Aho-Corasick automaton with frequent GO terms
    A = ahocorasick.Automaton()
    for term in frequent_go_terms:
        A.add_word(term, term)
    A.make_automaton()

    # Function to filter rows using Aho-Corasick
    def filter_frequent_terms(row):
        return A.exists(row['go_terms'])

    # Step 4: Filter the DataFrame using Aho-Corasick
    df_filtered = df[df.apply(filter_frequent_terms, axis=1)]

    # Step 4: Save the filtered DataFrame to a new file
    df_filtered.to_csv(output_file, index=False)

    # Step 5: Create a dictionary with the frequent GO terms and their assigned indices
    frequent_go_terms_dict = {term: idx for idx, term in enumerate(frequent_go_terms)}

    # Step 6: Save the dictionary to a file using pickle
    with open(go_terms_dict_file, 'wb') as f:
        pickle.dump(frequent_go_terms_dict, f)

    # Step 7: Print the total number of frequent terms
    print(f"Total number of frequent terms: {len(frequent_go_terms)}")

def filter_most_frequent(input_file = 'data/pretrain_go_frequent.csv', output_file = 'data/pretrain_go_frequent_8943.csv', go_terms_dict_file = 'data/8943_frequent_go_terms.pkl'):
    
    # Step 1: Read the DataFrame from the input file
    df = pd.read_csv(input_file)
    print("pretrain_go_frequent length:", len(df))
    # Step 2: Calculate the frequency of each GO term
    go_term_counts = df['go_terms'].value_counts()

    # Step 3: Identify the top 8,943 most frequent GO terms
    top_go_terms = go_term_counts.head(8943).index

    # Build the Aho-Corasick automaton with the top GO terms
    A = ahocorasick.Automaton()
    for term in top_go_terms:
        A.add_word(term, term)
    A.make_automaton()

    # Function to filter rows using Aho-Corasick
    def filter_top_terms(row):
        return A.exists(row['go_terms'])

    # Step 4: Filter the DataFrame using Aho-Corasick
    df_filtered = df[df.apply(filter_top_terms, axis=1)]

    # Step 5: Group by 'id' and merge 'go_terms' by ','
    df_grouped = df_filtered.groupby('id')['go_terms'].agg(lambda x: ','.join(x)).reset_index()
    print("len df_grouped:", len(df_grouped))

    # Step 6: Save the filtered DataFrame to a new file
    df_grouped.to_csv(output_file, index=False)

    # Step 7: Create a dictionary with the top GO terms and their assigned indices
    top_go_terms_dict = {term: idx for idx, term in enumerate(top_go_terms)}

    # Step 8: Save the dictionary to a file using pickle
    with open(go_terms_dict_file, 'wb') as f:
        pickle.dump(top_go_terms_dict, f)

    # Step 9: Print the total number of top terms
    print(f"Total number of top terms: {len(top_go_terms)}")


def merge(file1 = 'data/cluster-100-new/pretrain_ec.csv', file2 = 'data/pretrain_go_frequent_8943.csv'):
    
    # Step 1: Read the two CSV files into DataFrames
    df1 = pd.read_csv(file1, usecols=['id', 'seq'])
    df2 = pd.read_csv(file2)

    # Step 2: Merge the DataFrames on 'id' with a left join
    merged_df = pd.merge(df1, df2, on='id', how='left')

    # Step 3: Fill missing 'go_terms' with "-"
    merged_df['go_terms'].fillna("-", inplace=True)

    # Optional: Save the merged DataFrame to a new CSV file
    output_file = 'data/pretrain_go_final.csv'
    merged_df.to_csv(output_file, index=False)

    # Display the merged DataFrame
    print(len(merged_df))
    
    unique_ids_df1 = df1['id'].unique().tolist()
    unique_ids_df2 = df2['id'].unique().tolist()    
    all_ids_in_df2 = all(id in unique_ids_df1 for id in unique_ids_df2)

    # Print the result
    if all_ids_in_df2:
        print("All IDs in df2 are present in df1.")

merge()

