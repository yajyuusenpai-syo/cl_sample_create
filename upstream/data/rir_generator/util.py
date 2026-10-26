import csv

def load_csv(filename, as_dict=True):
    with open(filename, mode='r', encoding='utf-8') as file:
        if as_dict:
            reader = csv.DictReader(file)
        else:
            reader = csv.reader(file)
        return list(reader)

if __name__ == '__main__':
    data = load_csv("list.csv")
