import _pickle as pickle

def load():
    with open('test_batch', 'rb') as cifar10:
        data = pickle.load(cifar10, encoding='latin1')
        return data

datadict = load()
labels = datadict['labels'][0]
print(type(labels))
    # for index, content in enumerate(data):
        # print(content)
        # print(content['labels'])
# for index, name in enumerate(data):
#     print(name)
# print(label
