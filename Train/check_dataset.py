import train

# Checking Length of File
print("Length of the file: ", len(train.dataset_text), " characters")

# Printing Seperator
print("======================================================================")

# Printing first 250 Characters
print("Visualizing first 250 characters of Dataset: ")
print(train.dataset_text[:250])
print("======================================================================")