import tensorflow.contrib.keras as kr
import os




def read_vocab(vocab_dir):

    with open_file(vocab_dir) as fp:

        words = [_.strip() for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id 

def open_file(filename, mode='r'):
    return open(filename, mode, encoding='utf-8', errors='ignore')

def read_category():

    categories = ['\u4f53\u80b2', '\u8d22\u7ecf', '\u623f\u4ea7', '\u5bb6\u5c45', '\u6559\u80b2', '\u79d1\u6280', '\u65f6\u5c1a', '\u65f6\u653f', '\u6e38\u620f', '\u5a31\u4e50']

    categories = [x for x in categories]

    cat_to_id = dict(zip(categories, range(len(categories))))#
    return categories, cat_to_id


def read_file(filename):

    contents, labels = [], []
    with open_file(filename) as f:
        # print(filename)
        for line in f:
            try:
                label, content = line.strip().split('\t')
                if content:
                    contents.append(list(content))
                    labels.append(label)
            except:
                pass
    return contents, labels


def process_file(filename, word_to_id, cat_to_id, max_length=600):

    contents, labels = read_file(filename)

    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])


    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  
    return x_pad, y_pad


def to_category(content):

    categories, cat_to_id = read_category()
    return ''.join(categories[x] for x in content)