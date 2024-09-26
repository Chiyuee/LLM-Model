'''
Name: Chi Kwok
Student ID: W9023975

The project requires the following python files

main.py
TokenizerV2.py
Transformer.py
'''

# Libraries
from TokenizerV2 import Tokenizer as test_tokenizer
from Transformer import Transformer
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import pickle
import numpy as np
import pandas as pd
import sys
import time

from fastparquet import ParquetFile

def main():
    
    # Sets the device to an cuda-supported gpu, otherwise, use cpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Extracting data from file
    dataset = ParquetFile('Data\\archive\\a.parquet').to_pandas()
    x = dataset['title']
    y = dataset['text']

    token_set = None
    
    '''
    --- Settings ---
    These variables are adjustable to experiment model training
    
    Sentences used:
    
    The book of
    The writer for this article
    Nonsense sentence nonsense sentence nonsense sentence
    A & B High Performance Firearms
    '''
    
    data_length = len(x)
    batch_samples = 1500
    sentence_cap = 20
    batches = round(data_length / batch_samples)
    epochs = 10
    texts = ['The book of',
             'The writer of this article',
             'Nonsense sentence nonsense sentence nonsense sentence',
             'A and B High Performance Firearms']
    training = True

    # Instaniating tokenizer and giving sentence length cap
    test = test_tokenizer(sentence_cap = sentence_cap)
    
    # Capture tokens from data
    # Batch_cap_tokens stops loop at its integer (prevents overflow of VRAM)
    batch_cap_tokens = 20
    for batch in range(batches):
        if batch < batch_cap_tokens:
            start = batch * batch_samples
            x_batch = x[start : start + batch_samples]
            y_batch = y[start : start + batch_samples]
            test.Feed(x_batch, y_batch)
        else: break
    
    # Saving token file
    if training:
        with open('Tokens.pkl', 'wb') as file:
            pickle.dump(list(test.tokens), file)
    
    test.Token_Info()
    token_length = test.Get_Token_length()
    pad_index = test.Get_Pad_index()
    
    # Transformer model
    model = Transformer(token_length, token_length, pad_index, pad_index, embed_size = 256).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index = 0)
    optimizer = Adam(model.parameters(), lr = 0.001)

    # Training/Evaluation Process
    if training:
        
        # batches = 300
        model.train()
        
        now = time.time()
        
        # Looping through epochs of data for training
        for epoch in range(epochs):
            for batch in range(1, batches + 1):
                section = epoch * batch
                inputs = x[section: section + batches]
                target = y[section: section + batches]
                
                inputs, target = test.Feed(inputs, target)
                inputs, target = test.Extract(x = inputs, y = target)
                
                inputs = torch.tensor(inputs).to(device)
                target = torch.tensor(target).to(device)
                
                optimizer.zero_grad()

                output = model(inputs, target[:, :-1])
                
                # output = output.view(-1, token_length)
                
                output = output.view(-1, output.size(-1))
                target = target[:, 1:].contiguous().view(-1) 
                
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()
                
                loss = round(loss.item(), 3)
                
                line = f'\rEpoch: {epoch + 1}/{epochs} Batch: {batch}/{batches} Loss: {loss}'
                sys.stdout.write(line)
                sys.stdout.flush()
            
            print('')
            print('------------------------------------')
            
            # Save state of model through each epoch
            torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')

        later = time.time()
        time_result = later - now
        print(f'Seconds of training: {time_result}')
        
    else:
        # Loads previous models
        model.load_state_dict(torch.load('model_epoch_1.pth'))
        with open('Tokens.pkl', 'rb') as file:
            token_set = pickle.load(file)
        
    print('')
    print('------------------------------------')
    print('Starting Evaluation')
    print('------------------------------------')
    
    # STARTING MODEL EVALUATION
    model.eval()
    
    for text in texts:
    
        x_text = pd.Series([text])
        y_text = pd.Series(['<PAD>'])
        
        x_test, y_test = test.Feed(x_text, y_text)
        x_test, y_test = test.Extract(x_test, y_test)
        
        x_test = torch.tensor(x_test).to(device)
        y_test = torch.tensor(y_test).to(device)
        
        prediction = model(x_test, y_test[:, :-1])
        
        # Appending predicted word to list
        prediction = prediction.tolist()
        prediction = np.argmax(prediction[0][0])
        prediction = test.Translate([prediction], token_set)
        prediction = " " + prediction[0]
        x_text = x_text.add(prediction) 
        
        # Will only run through prediction if predicted word is not EOS
        for i in range(1, sentence_cap):
            if prediction != ' <EOS>':
                
                x_test, y_test = test.Feed(x_text, y_text)
                x_test, y_test = test.Extract(x_test, y_test)
                
                x_test = torch.tensor(x_test).to(device)
                y_test = torch.tensor(y_test).to(device)

                prediction = model(x_test, y_test[:, :-1])
                prediction = prediction.tolist()
                
                prediction = np.argmax(prediction[0][i])
                prediction = test.Translate([prediction], token_set)
                prediction = " " + prediction[0]
                x_text = x_text.add(prediction)
                y_text = y_text.add(prediction)
            else:
                print("ENDED BY EOS")
                break
        print(x_text[0])

    test.Token_Info()

if __name__ == '__main__': main()