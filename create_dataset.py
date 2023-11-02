import os
import string
import xml.etree.ElementTree as ET
import re

from nltk import word_tokenize
from nltk.corpus import stopwords
from openpyxl import Workbook
from tqdm import tqdm
stop = stopwords.words('english')

def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text.replace('-', ' ').replace('/',' '))
    tokens = [w for w in tokens if len(w)>2]
    table = str.maketrans(' ', ' ', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in stripped if word.isalpha() and word not in stop]
    clean_text = ' '.join(words)
    return clean_text

class Dataset:
    def __init__(self, path):
        self.path = path
        self.instance_map = {}

    def generate(self):
        for root, dirs, files in os.walk(self.path):
            for file in tqdm(files, desc=f'Reading dataset: {self.path}'):
                instance_map = self.XMLReader(os.path.join(root, file))
        return instance_map

    def XMLReader(self, file_path):
        tree = ET.parse(file_path)
        root = tree.getroot()

        sentences = root.findall('sentence')
        instance_map = {}

        for sentence in sentences:
            entities = sentence.findall('entity')
            ddis = sentence.findall('ddi')
            entity_map = {}
            skip_sentence = False  # Flag to skip the current sentence if an exception is caught
            for entity in entities:
                id = entity.get('id')
                name = entity.get('text')
                entity_map[id] = name

            for id1, drug1 in entity_map.items():
                drug1 = drug1.split(",")[0]
                for id2, drug2 in entity_map.items():
                    sentence_text = sentence.get('text')
                    label = "FALSE"
                    type = "null"
                    drug2 = drug2.split(",")[0]
                    if id1 != id2:
                        for id3, name3 in entity_map.items():
                            if id1 != id3 and id2 != id3:
                                try:
                                    sentence_text = sentence_text.replace(name3, "OTHER_DRUG")
                                except:
                                    skip_sentence = True  # Set the flag to skip this sentence
                                    break  # Break out of the inner loop

                            if skip_sentence:
                                break  # Break out of the outer loop

                        for ddi in ddis:
                            e1 = ddi.get('e1')
                            e2 = ddi.get('e2')
                            if id1 == e1 and id2 == e2:
                                label = "TRUE"
                                type = ddi.get('type')

                        instance = [drug1[0].upper()+drug1[1:], drug2[0].upper()+drug2[1:], label, type]
                        sentence_text = preprocess(sentence_text)

                        if sentence_text not in self.instance_map and len(drug1) > 2 and len(drug2) > 2:
                            self.instance_map[sentence_text] = instance

        return self.instance_map

if __name__ == "__main__":
    folder = os.path.join(os.getcwd(), 'dataset/')
    datasets = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]

    instance_maps = list()

    for dataset in datasets:
        path = os.path.join('dataset/', dataset)
        dataset_obj = Dataset(path)
        instance_map = dataset_obj.generate()
        instance_maps.append(instance_map)

    workbook = Workbook()
    worksheet = workbook.active
    worksheet.append(["normalized_sentence", "drug1", "drug2", "label", "type"])

    for instance_map in instance_maps:
        for sentence, other_features in instance_map.items():
            row = [sentence.replace("\n|\r", " ").replace('\"', "")] + other_features
            worksheet.append(row)

    workbook.save("dataset/DDIdataset-en.xlsx")
