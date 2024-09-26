from fastparquet import ParquetFile
import string

'''
I've decided to build my own tokenizer for more flexibility during training
and research. One of the key findings was removing punctuation between sentences
in data has reduced the amount of redundant tokens created, helping in optimising VRAM usage
and training with more popular tokens

Through my studies, I've found there are numerous ways to tokenize your data. Popular models 
such as GPT models separate letters together, whereas for this tokenizer, I decided to tokenize
whole words. The decision made was to reduce the amount of VRAM usage.

V2 of the tokenizer has cleaner code and includes padding whereas the first tokenizer did not 
include padding. 
'''

class Tokenizer():
    def __init__(self, sentence_cap = 20):
        self.tokens = set(['<PAD>', '<EOS>', '<POS>', '<FILL>'])
        self.cap = sentence_cap
        self.processed_x = []
        self.processed_y = []
        self.x_length = 0
        self.y_length = 0
    
    # Collect tokens and add 
    def Feed(self, x_data, y_data):
        x = x_data
        y = y_data
        
        processed_x = []
        processed_y = []
        
        # Preprocessing stage
        if x.any() == None or y.any() == None: print("No data detected")
        else:
            
            # x is the input data and y is the target data
            for words in x:
                sentence = ['<POS>']
                words = words.split()
                for word in words[:self.cap]:
                    new_string = ''
                    
                    # Checks for any punctuation in the sentence and removes it
                    # Proven to reduce redundant tokens and filling in more popular tokens
                    for char in word:
                        if char not in string.punctuation: 
                            new_string += char
                    
                    # Checks to see if the string created is not empty, and only 
                    # then will it add to the token to the set
                    if new_string != '':
                        self.tokens.add(new_string.lower())
                        sentence.append(new_string.lower())
                            
                # Append EOS at the end of the sentence
                sentence.append('<EOS>')
                processed_x.append(sentence)
                if len(sentence) > self.x_length:
                    self.x_length = len(sentence)
            
            # Same preprocessing as input data but with target data
            for words in y:
                sentence = ['<POS>']
                words = words.split()
                for word in words[:self.cap]:
                    new_string = ''
                    for char in word:
                        if char not in string.punctuation: 
                            new_string += char
                    
                    if new_string != '':
                        self.tokens.add(new_string.lower())
                        sentence.append(new_string.lower())
                
                sentence.append('<EOS>')
                processed_y.append(sentence)
                if len(sentence) > self.y_length:
                    self.y_length = len(sentence)
        
        return processed_x, processed_y
    
    # This will pad and convert words into tokens
    def Extract(self, x, y):
        tokens = list(self.tokens)
        
        input_data = []
        target_data = []
        
        for sample in x:
            tokenized_sample = []
            for word in sample:
                token_index = None
                try:
                    token_index = tokens.index(word)
                except:
                    
                    # If the word is not included in the set, it wil replace the word
                    # as FILL. The purpose is to continue the transformer process
                    # without the code giving an error. Ideally, this should not be
                    # an issue for Large Language models, but due to the contraints
                    # of having 12GB of VRAM, a work around must be made to continue
                    # this research
                    print(f'X Word {word} is not tokenized. Replacing with <FILL>')
                    token_index = tokens.index('<FILL>')
                tokenized_sample.append(token_index)
            
            # Calculating the length of PADDING required and appending it
            sample_length = len(sample)
            padding_length = self.x_length - sample_length
            tokenized_sample = tokenized_sample + ([tokens.index('<PAD>')] * padding_length)
            
            input_data.append(tokenized_sample)
        
        # Identical process as X extraction
        for sample in y:
            tokenized_sample = []
            for word in sample:
                token_index = None
                try:
                    token_index = tokens.index(word)
                except:
                    print(f'Y Word {word} is not tokenized. Replacing with <FILL>')
                    token_index = tokens.index('<FILL>')
                tokenized_sample.append(token_index)
            
            sample_length = len(sample)
            padding_length = self.y_length - sample_length
            tokenized_sample = tokenized_sample + ([tokens.index('<PAD>')] * padding_length)
            
            target_data.append(tokenized_sample)
            
        return input_data, target_data
    
    # Function gives key information about the tokenizer built
    def Token_Info(self):
        
        tokens_list = list(self.tokens)
        token_length = len(self.tokens)
        pos = tokens_list.index('<POS>')
        eos = tokens_list.index('<EOS>')
        pad = tokens_list.index('<PAD>')
        fill = tokens_list.index('<FILL>')
        
        print('------------------------------------')
        print(f'Token Length: {token_length}')
        print(f'POS Token: {pos}')
        print(f'EOS Token: {eos}')
        print(f'PAD Token: {pad}')
        print(f'FILL Token: {fill}')
        print('------------------------------------')
    
    def Translate(self, sequence, token_file = None):
        
        if token_file == None:
            print('Token file is empty, replacing with currently trained tokens')
            tokens = list(self.tokens)
        else:
            tokens = token_file
        translation = []
        for token in sequence:
            translation.append(tokens[token])
        return translation
    
    # Translate the token number to English
    def Get_Token_length(self):
        return len(self.tokens)
    
    def Get_Pad_index(self):
        tokens = list(self.tokens)
        return tokens.index('<PAD>')
    
    def Get_EOS_index(self):
        tokens = list(self.tokens)
        return tokens.index('<EOS>')