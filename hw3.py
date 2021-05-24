import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator
import spacy
import numpy as np
import random
import math
import time

###python -m spacy download en
###python -m spacy download jp

# lang tonenizer modeling 
spacy_de = spacy.load('jp')
spacy_en = spacy.load('en')

def tokenize_de(text):
    # 일어 tokenize해서 단어들을 리스트로 만든 후 reverse 
    return [tok.text for tok in spacy_jp.tokenizer(text)][::-1]
    
def tokenize_en(text):
    # 영어 tokenize해서 단어들을 리스트로 만들기
    return [tok.text for tok in spacy_en.tokenizer(text)]

# SRC = source = input
SRC = Field(tokenize = tokenize_jp, init_token='<sos>', eos_token='<eos>', lower=True)
# TRG = target = output
TRG = Field(tokenize = tokenize_en, init_token='<sos>', eos_token='<eos>', lower=True)

train_data, valid_data, test_data = Multi30k.splits(exts=('.jp', '.en'), fields=(SRC,TRG))

##print(vars(train_data.examples[0]))

SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)

batch_size = 128

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(train_data, valid_data, test_data), batch_size=batch_size)


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self. hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        # src = [src len, batch_size]
        embedded = self.dropout(self.embedding(src))
        
        # embedded = [src len, batch size, emb dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        
        # outputs = [src len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]
        
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout_
        
    def forward(self, input, hidden, cell):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]
        # Decoder에서 항상 n directions = 1
        # 따라서 hidden = [n layers, batch size, hid dim]
        # context = [n layers, batch size, hid dim]
        
        # input = [1, batch size]
        input = input.unsqueeze(0)
        
        # embedded = [1, batch size, emb dim]
        embedded = self.dropout(self.embedding(input))
        
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        
        # output = [seq len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]
        
        # Decoder에서 항상 seq len = n directions = 1 
        # 한 번에 한 토큰씩만 디코딩하므로 seq len = 1
        # 따라서 output = [1, batch size, hid dim]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]
        
        # prediction = [batch size, output dim]
        prediction = self.fc_out(output.squeeze(0))
        
        return prediction, hidden, cell
        
class Seq2Seq(nn.Module):
   def __init__(self, encoder, decoder, device):
       super().__init__()
       
       self.encoder = encoder
       self.decoder = decoder
       self.device = device
       
       # Encoder와 Decoder의 hidden dim이 같아야 함
       assert encoder.hid_dim == decoder.hid_dim
       # Encoder와 Decoder의 layer 개수가 같아야 함
       assert encoder.n_layers == decoder.n_layers
       
   def forward(self, src, trg, teacher_forcing_ratio=0.5):
       # src = [src len, batch size]
       # trg = [trg len, batch size]
       
       trg_len = trg.shape[0]
       batch_size = trg.shape[1]
       trg_vocab_size = self.decoder.ouput_dim
       
       # decoder 결과를 저장할 텐서
       outputs = torch.zeros(trg_len, batch_size, trg_vocab_size)
       
       # Encoder의 마지막 은닉 상태가 Decoder의 초기 은닉상태로 쓰임
       hidden, cell = self.encoder(src)
       
       # Decoder에 들어갈 첫 input은 <sos> 토큰
       input = trg[0, :]
       
       # target length만큼 반복
       # range(0,trg_len)이 아니라 range(1,trg_len)인 이유 : 0번째 trg는 항상 <sos>라서 그에 대한 output도 항상 0 
       for t in range(1, trg_len):
           output, hidden, cell = self.decoder(input, hidden, cell)
           outputs[t] = output
           
           # random.random() : [0,1] 사이 랜덤한 숫자 
           # 랜덤 숫자가 teacher_forcing_ratio보다 작으면 True니까 teacher_force=1
           teacher_force = random.random() < teacher_forcing_ratio
           
           # 확률 가장 높게 예측한 토큰
           top1 = output.argmax(1) 
           
           # techer_force = 1 = True이면 trg[t]를 아니면 top1을 input으로 사용
           input = trg[t] if teacher_force else top1
       
       return outputs

input_dim = len(SRC.vocab)
output_dim = len(TRG.vocab)

# Encoder embedding dim
enc_emb_dim = 256
# Decoder embedding dim
dec_emb_dim = 256

hid_dim=512
n_layers=2

enc_dropout = 0.5
dec_dropout=0.5

enc = Encoder(input_dim, enc_emb_dim, hid_dim, n_layers, enc_dropout)
dec = Decoder(output_dim, dec_emb_dim, hid_dim, n_layer, dec_dropout)

model = Seq2Seq(enc, dec, device)

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uiform_(param.data, -0.08, 0.08)

model.apply(init_weights)

optimizer = optim.Adam(model.parameters())

# <pad> 토큰의 index를 넘겨 받으면 오차 계산하지 않고 ignore하기
# <pad> = padding
trg_pad_idx = TRG.vocab.stoi[TRG.pad_token]

criterion = nn.CrossEntropyLoss(ignore_index = trg_pad_idx)

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss=0
    
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg
        
        optimizer.zero_grad()
        
        output = model(src, trg)
        
        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]
        output_dim = output.shape[-1]
        
        # loss 함수는 2d input으로만 계산 가능 
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        
        # trg = [(trg len-1) * batch size]
        # output = [(trg len-1) * batch size, output dim)]
        loss = criterion(output, trg)
        
        loss.backward()
        
        # 기울기 폭발 막기 위해 clip
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss+=loss.item()
        
    return epoch_loss/len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            
            # teacher_forcing_ratio = 0 (아무것도 알려주면 안 됨)
            output = model(src, trg, 0)
            
            # trg = [trg len, batch size]
            # output = [trg len, batch size, output dim]
            output_dim = output.shape[-1]
            
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            
            # trg = [(trg len - 1) * batch size]
            # output = [(trg len - 1) * batch size, output dim]
            
            loss = criterion(output, trg)
            
            epoch_loss+=loss.item()
        
        return epoch_loss/len(iterator)

N_EPOCHS = 10
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    
    start_time = time.time()
    
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut1-model.pt')
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

###que (model)####
model.load_state_dict(torch.load('tut1-model.pt'))

test_loss = evaluate(model, test_iterator, criterion)

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
            
