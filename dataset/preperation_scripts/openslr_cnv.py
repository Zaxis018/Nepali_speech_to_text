import csv

# Define the input TSV file and the output CSV file
input_tsv_file = 'line_index.tsv'
output_csv_file = 'openslr-tts.csv'

# Define the string to add to the first column
string_to_add = 'wavs/'

# Open the TSV file for reading
with open(input_tsv_file, mode='r', newline='', encoding='utf-8') as tsv_file:
    tsv_reader = csv.reader(tsv_file, delimiter='\t')
    
    # Open the CSV file for writing
    with open(output_csv_file, mode='w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        
        # Loop through the rows in the TSV file
        for row in tsv_reader:
            # Add the string to the first column
            row[0] = string_to_add + row[0] + '.wav'
            # Write the modified row to the CSV file
            csv_writer.writerow(row)

print(f"Modified data has been written to {output_csv_file}")
