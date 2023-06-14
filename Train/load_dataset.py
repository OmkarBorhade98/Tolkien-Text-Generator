# Mention path to the .txt file containing Data
path_to_file = '../Dataset/dataset.txt'

# Load Dataset
dataset_text = open(path_to_file, mode ="r", encoding="utf-8").read()
print("Number of characters in dataset: ", len(dataset_text))

# Lets Check Number of Unique Characters :)
# set(): Builds unordered set of unique elements
#sorted(): Sorts the input
vocab = sorted(set(dataset_text))
print("Number of unique characters: ", len(vocab))

# Currate input and target dataset
def split_input_target(sequence):
    input_dataset = sequence[:-1]
    target_dataset = sequence[1:]
    return input_dataset, target_dataset