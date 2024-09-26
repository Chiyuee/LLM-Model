from fastparquet import ParquetFile
import pandas as pd
import string 

class Tokenizer():
    def __init__(self, data = None, stop = 20):
        
        self.stop = stop
        self.tokens = set()
        self.index = dict()
        self.all_words = dict()
        self.tokens.update(['<PAD>', '<EOS>', '<POS>'])
        
        if data != None: self.Feed_Data(data)
        
    def Feed_Data(self, data):
        dataset = ParquetFile(data).to_pandas()
        dataset = dataset.drop(['id', 'categories'], axis = 1)
        
        stop = self.stop
        for enum, text in enumerate(dataset['text']):
            if enum > stop: break
            
            words = text.split()
            
            for word in words:
                
                if any(char not in string.punctuation for char in word):
                    self.tokens.add(word.lower())
                    
                #else: self.tokens.add(word.lower()) # DELETE LATER
                    
        for enum, text in enumerate(dataset['title']):
            if enum > stop: break
            
            words = text.split()
            
            for word in words:
                
                if any(char not in string.punctuation for char in word):
                    self.tokens.add(word.lower())
                    
                #else: self.tokens.add(word.lower()) # DELETE LATER
                
        for i in string.punctuation:
            self.tokens.add(i)
        
        self.index = dict((i,j) for i,j in enumerate(self.tokens))
        self.all_words = dict((j,i) for i,j in enumerate(self.tokens))
        # print(self.index)
        
    def Describe(self):
        print(f'Set contains: {len(self.tokens)}')
        return len(self.tokens)
    
    def Get_Words(self):
        return self.all_words
    
    def Find(self, sequence):
        words_list = []
        for words in sequence:
            print(f'Sequence: {sequence}')
            for word in words:
                print(f'Token: {word}')
                words_list.append(self.index[word])
                print('Word list: {words_list}')
            
        return words_list
            
