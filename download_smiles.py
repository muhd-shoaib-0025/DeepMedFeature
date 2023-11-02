import threading
import time
from multiprocessing import freeze_support

import pandas as pd
import pubchempy as pcp
from tqdm import tqdm
import multiprocessing as mp

def split_list(lst, parts):
    part_length = len(lst) // parts
    for i in range(0, len(lst), part_length):
        yield lst[i:i + part_length]

thread_id_counter = 0

def download_smiles(drugs, smiles):
    global thread_id_counter
    thread_id = thread_id_counter
    thread_id_counter += 1
    success=0
    failure=0
    for i, drug in enumerate(drugs):
        flag = False
        print('Reading data ... #rows=' + str(len(drugs)) +
              '; i: ' + str(i) +
              '; thread_id: ' + str(thread_id) +
              '; success: ' + str(success) +
              '; failure: ' + str(failure))
        for i in range(0, 5):
            try:
                time.sleep(0.1)
                smile = pcp.get_compounds(drug, 'name')
                if smile:
                    smile = smile[0].isomeric_smiles
                    smiles[drug] = smile
                flag = True
                break
            except:
                flag = False
                continue
        if flag:
            success+=1
        else:
            failure+=1

if __name__ == '__main__':
    freeze_support()

    df = pd.read_excel('dataset/DDIdataset-' + 'en' + '.xlsx', engine='openpyxl', sheet_name='Sheet')['drug1'].tolist()
    df2 = pd.read_excel('dataset/DDIdataset-' + 'en' + '.xlsx', engine='openpyxl', sheet_name='Sheet')['drug2'].tolist()
    drug_array = split_list(list(set(df + df2)), mp.cpu_count())

    total_drugs = len(set(df + df2))
    print('#total_drugs', total_drugs)

    threads = []
    smiles = mp.Manager().dict()

    for array in drug_array:
        t = threading.Thread(target=download_smiles, args=(array, smiles, ))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    f = open('pubchem_smiles.csv', 'w')
    f.write('drug, smile\n')
    for key, value in smiles.items():
        f.write(f'{key}, {value}\n')
    f.close()

    total_drugs_downloaded = len(smiles)

    print('#total_drugs_downloaded', total_drugs_downloaded)


