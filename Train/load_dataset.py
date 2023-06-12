# Mention path to the .txt file containing Data
path_to_file = '../Dataset/dataset.txt'

# Load Dataset
dataset_text = open(path_to_file, mode ="r", encoding="utf-8").read()

# Lets Check Number of Unique Characters :)
# set(): Builds unordered set of unique elements
#sorted(): Sorts the input
vocab = sorted(set(dataset_text))
print("Number of unique characters: ", len(vocab))