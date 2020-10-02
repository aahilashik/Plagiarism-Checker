from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import LancasterStemmer 
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import string
import pandas as pd
import numpy as np


class Plagiarism_Checker:
    
    def __init__(self, algorithm="TFIDF"):
        
        self.algorithm = algorithm
        self.stopWords = stopwords.words('english')
        
        self.wsTok      = WhitespaceTokenizer() 
        self.stemmer    = LancasterStemmer()
        self.countVect  = CountVectorizer()
        self.tfidfVect  = TfidfVectorizer()
        
        self.queryData = []
        self.srcData   = []
        
        
        
    def preprocess(self, documents):
        
        processed = []
        for document in documents:
            
            #1 Removing Punctuations
            data = document.translate(str.maketrans("'", " ", string.punctuation))
                        
            #2 Converting to Lowercase            
            data = data.lower()
            
            #3 Tokenization
            data = self.wsTok.tokenize(data)
                        
            #4 Removing Stop Words            
            data = [word for word in data if not word in self.stopWords]
            
            #5 Stemming words            
            data = [self.stemmer.stem(word) for word in data]

            processed.append(data)
        
        return processed
        
        
    
    def setQueryText(self, data, clearData=True):
        
        if type(data) != list:
            print("Error : Set Query - Datatype should be 'list'")
        
        if clearData:
            self.queryData = []
            
        for d in data:
            self.queryData.append(d)
        
        
    
    def setSourceText(self, data, clearData=True):
        
        if type(data) != list:
            print("Error : Set Source - Datatype should be 'list'")
        
        if clearData:
            self.srcData = []
            
        for d in data:
            self.srcData.append(d)
        
    
        
    def jaccardSimilarity(self, query, document):
        
        inter_l = list(set(query) &  set(document))
        union_l = list(set(query) or set(document))
        
        return len(inter_l)/len(union_l)



    def getPlagMatrix(self, documents):
        
        if self.algorithm == "TFIDF":
            data    = [','.join(str(v) for v in document) for document in documents]
            tfidf   = self.tfidfVect.fit_transform(data)
            similarityMatrix = cosine_similarity(tfidf)
            
        
        elif self.algorithm == "TF":
            data = [','.join(str(v) for v in document) for document in documents]
            sparse_matrix    = self.countVect.fit_transform(data)
            doc_term_matrix  = sparse_matrix.todense()
            tf = pd.DataFrame(doc_term_matrix, columns=self.countVect.get_feature_names())
            similarityMatrix = cosine_similarity(tf)
        
        else: 
            similarityMatrix = np.zeros((len(documents), len(documents)))
            for i, doc1 in enumerate(documents):
                for j, doc2 in enumerate(documents):
                    similarityMatrix[i][j] = self.jaccardSimilarity(doc1, doc2)
        
        return similarityMatrix                    

                    
                    
    def getReport(self):
        
        query   = self.preprocess(self.queryData)
        src     = self.preprocess(self.srcData)
        similarity = []
        
        for q in query:
        
            documents = [q] + src
            sim = self.getPlagMatrix(documents)[0][1:] 
            similarity.append(sim)
            
        return similarity
                    
                    
                    
         






































           
   
"""                 
                    
                    
      

file1 = "Z:/Ashik/Ashik_Resume.docx"
file2 = "Z:/Azhar/Azhar_Resume.docx"

doc1 = docx2txt.process(file1)
doc2 = docx2txt.process(file2)

checker = Plagiarism_Checker("TF")

checker.setQueryText([doc1])

checker.setSourceText([doc2, doc2, doc2, doc2])

q=checker.getReport()              

                    
                    
                    
             
"""       
                    