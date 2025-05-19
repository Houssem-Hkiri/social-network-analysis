# Convert the edge list to CSV for Gephi
input_file = "facebook_combined.txt"
output_file = "facebook_edges.csv"

with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    outfile.write("Source,Target\n")
    for line in infile:
        if line.strip():  # skip empty lines
            source, target = line.strip().split()
            outfile.write(f"{source},{target}\n")
print('Done')
