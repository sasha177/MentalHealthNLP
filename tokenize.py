import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import LancasterStemmer


#need to make a table with subject ID, diagnosis (from admissions), admission times, mental status (from chart events), HADM_ID using merge in pandas
#print (notes_file.loc[notes_file['SUBJECT_ID'] == 27799])

"""
medications:
fluoxetine: 109
CHLORPROMAZINE: 10
CITALOPRAM: 311

illnesses:
SCHIZOPHRENIA: 26
DEPRESSION: 34
ANXIETY: 27
OCD: 25
ADHD: 25

patients
27799: major depression, alcohol withdrawal, doctor told him to take haldol for agitation, diazepam for anxiety
"""

part_file = pd.read_csv("/Users/sashamittal/Downloads/SERVICES.csv")

#mofidying some settings so that more of the data actually shows up

pd.options.display.max_rows = 50
pd.options.display.max_columns = 10
pd.set_option('max_info_columns', 11)
pd.set_option('max_colwidth', 4000)

lst = LancasterStemmer()

#stems depressed to depress
#stems anxiety to anxy

final_set = []
def tokenize_and_stem(x):
    y=0
    for paragraph in x:  # does the tokenizing
        paragraph_set = []
        paragraph = str(paragraph)
        paragraph = word_tokenize(paragraph)
        for word in paragraph:
            word = lst.stem(word)
            word = str(word)
            paragraph_set.append(word)
        final_set.append(paragraph_set)
        y = y + 1
        print(y)

tokenize_and_stem(part_file['TEXT'])

csv_input = pd.read_csv(PART FILE PATHNAME)
csv_input['RESULT'] = final_set
csv_input.to_csv('output.csv', index=False) #new csv file is made

output = pd.read_csv("output.csv") #new csv file in named output


print(output[output['RESULT'].str.contains('schizophrenia')]) #prints rows based on condition

