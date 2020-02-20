import glob, os

# Current directory
current_dir = os.path.dirname(os.path.abspath(__file__))+"/monkey_data_new"


#current_dir = os.getcwd()+"/Images/duck"
print(current_dir)

#current_dir = 'Your dataset path.'

# Directory where the data will reside, relative to 'darknet.exe'
#path_data = './NFPAdataset/'

# Percentage of images to be used for the test set
percentage_test = 10;

# Create and/or truncate train.txt and test.txt
file_train = open('monkey_train.txt', 'w')
file_test = open('monkey_test.txt', 'w')

# Populate train.txt and test.txt
counter = 1
count_test = 0
count_train = 0
index_test = round(100 / percentage_test)
for pathAndFilename in glob.iglob(os.path.join(current_dir, "*.jpg")):
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))

    if counter == index_test:
        counter = 1
        count_test = count_test + 1
        file_test.write(current_dir + "/" + title + '.jpg' + "\n")
    else:
        file_train.write(current_dir + "/" + title + '.jpg' + "\n")
        counter = counter + 1
        count_train = count_train + 1
print("count_test: %d\tcount_train: %d"%(count_test, count_train))
print("total: %d\n"%(count_test+count_train))
