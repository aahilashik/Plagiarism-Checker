################ Demo 1 ################

from plagiarism_checker import Plagiarism_Checker
from bs4 import BeautifulSoup 
from urllib.request import urlopen


qryUrls = ["https://en.wikipedia.org/wiki/Natural_language_processing", "https://en.wikipedia.org/wiki/Jaccard_index"]

srcUrls = ["https://en.wikipedia.org/wiki/Jaccard_index", "https://en.wikipedia.org/wiki/Tf%E2%80%93idf#Mathematical_details"]

           
           
sources = []
queries = []

#https://github.com/vipulaggarwal92/Detecting-Plagiarism-in-School-Assignments-Python-NLTK-/blob/master/Document%20Similarity%20Using%20NLTK%20in%20Python.ipynb
#https://stackoverflow.com/questions/328356/extracting-text-from-html-file-using-python

for url in qryUrls:
    html = urlopen(url).read()
    soup = BeautifulSoup(html, features="html.parser")
    
    for script in soup(["script", "style"]):
        script.extract()
    
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if len(chunk)>30)
    
    queries.append(text)

for url in srcUrls:
    html = urlopen(url).read()
    soup = BeautifulSoup(html, features="html.parser")
    
    for script in soup(["script", "style"]):
        script.extract()
    
    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if len(chunk)>30)
    
    sources.append(text)
    


checker = Plagiarism_Checker("TF")

checker.setQueryText(queries)

checker.setSourceText(sources)

sim = checker.getReport()  













################ Demo 2 ################

import docx2txt
from plagiarism_checker import Plagiarism_Checker

file1 = "Z:/Ashik/doc_demo1.docx"
file2 = "Z:/Azhar/doc_demo2.docx"

ms = docx2txt.process(file1)
doc1 = docx2txt.process(file2)

checker = Plagiarism_Checker("TF")

checker.setQueryText([ms])

checker.setSourceText([doc1])

sim = checker.getReport()

