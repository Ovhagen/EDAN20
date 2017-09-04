import regex as re
import pickle
import codecs
import os
import sys


def words(text):
    return re.finditer(r'\p{L}+', text)

def get_files(dir, suffix):
    """
    Returns all the files in a folder ending with suffix
    :param dir:
    :param suffix:
    :return: the list of file names
    """
    files = []
    for file in os.listdir(dir):
        if file.endswith(suffix):
            files.append(file)
    return files

def words_to_index(ws):
    word_idx = {}
    for match in ws:
        word = match.group()
        ind = match.start()
        if (word_idx.get(word)):
            word_idx[word].append(ind)
        else:
            word_idx[word] = [ind]
    return word_idx

def fileindex(dir):
    master_index = {}
    files = get_files(dir, ".txt")
    for file in files:
        text = open(dir+file).read().lower().strip()
        ws = words(text)
        idx = words_to_index(ws)
        for word in idx:
            if word in master_index:
                master_index[word][file] = idx[word]
            else:
                master_index[word] = {}
                master_index[word][file] = idx[word]

    pickle.dump(master_index, open("master_index" + ".idx", "wb")) #[:-4] to remove .txt part of string


#Main
arguments = sys.argv
if arguments[1] != '':
    print("Search in folder", arguments[1])
    fileindex(arguments[1]+'/') #Starts generating indexed dictionary from files
else:
    raise ValueError("Invalid argument. Input should be the directory from where files should be read.")


#Tests
dic = pickle.load(open('master_index'+".idx", "rb"))
print('gjord:', dic['gjord']) #Should be [8551, 183692, 220875]
#print(dic['uppklarnade']) #Should be [8567]
print('stjärnor:', dic['stjärnor']) #Should be [8590]
