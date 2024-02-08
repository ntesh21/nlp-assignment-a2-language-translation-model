import random
import torch
import torch.nn as nn
import torch.nn.functional as F




class Seq2SeqPackedAttention(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.device  = device
        
    def create_mask(self, src):
        mask = (src == self.src_pad_idx).permute(1, 0) 
        return mask
        
    def forward(self, src, src_len, trg, teacher_forcing_ratio = 0.5):
        batch_size = src.shape[1]
        trg_len    = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        outputs    = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        attentions = torch.zeros(trg_len, batch_size, src.shape[0]).to(self.device)
        encoder_outputs, hidden = self.encoder(src, src_len)
        input_ = trg[0, :]
        mask   = self.create_mask(src)
        for t in range(1, trg_len):
            #send them to the decoder
            output, hidden, attention = self.decoder(input_, hidden, encoder_outputs, mask)
            #append the output to a list
            outputs[t] = output
            attentions[t] = attention
            teacher_force = random.random() < teacher_forcing_ratio
            top1          = output.argmax(1)  #autoregressive
            input_ = trg[t] if teacher_force else top1
            
        return outputs, attentions



class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn       = nn.GRU(emb_dim, hid_dim, bidirectional=True)
        self.fc        = nn.Linear(hid_dim * 2, hid_dim)
        self.dropout   = nn.Dropout(dropout)
        
    def forward(self, src, src_len):
        #embedding
        embedded = self.dropout(self.embedding(src))
        #packed
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len.to('cpu'), enforce_sorted=False)
        #rnn
        packed_outputs, hidden = self.rnn(packed_embedded)
        #unpacked
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)
        #-1, -2 hidden state
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim = 1)))  
        return outputs, hidden
    

class Attention(nn.Module):
    def __init__(self, hid_dim, attention_type='general'):
        super().__init__()
        self.attention_type = attention_type
        if attention_type == 'general':
            self.v = nn.Linear(hid_dim, 1, bias=False)
        elif attention_type == 'multiplicative':
            self.W = nn.Linear(hid_dim, hid_dim)
        elif attention_type == 'additive':
            self.W1 = nn.Linear(hid_dim, hid_dim)
            self.W2 = nn.Linear(hid_dim, hid_dim)
            self.v = nn.Linear(hid_dim, 1, bias=False)
        else:
            raise ValueError("Invalid attention_type. Choose from 'general', 'multiplicative', or 'additive'.")

        self.U = nn.Linear(hid_dim * 2, hid_dim)

    def forward(self, hidden, encoder_outputs, mask):
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        if self.attention_type == 'general':
            energy = self.v(torch.tanh(self.U(encoder_outputs))).squeeze(2)
        elif self.attention_type == 'multiplicative':
            # import pdb;pdb.set_trace()
            # print(hidden.size())
            # print(self.W(encoder_outputs).permute(0, 2, 1).size())
            # query @ self.W @ values.T 
            energy = torch.matmul(hidden, self.W(encoder_outputs).permute(0, 2, 1))
        elif self.attention_type == 'additive':
            energy = self.v(torch.tanh(self.W1(hidden) + self.W2(encoder_outputs))).squeeze(2)
        else:
            raise ValueError("Invalid attention_type. Choose from 'general', 'multiplicative', or 'additive'.")

        energy = energy.masked_fill(mask, -1e10)

        return F.softmax(energy, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention  = attention
        self.embedding  = nn.Embedding(output_dim, emb_dim)
        self.rnn        = nn.GRU((hid_dim * 2) + emb_dim, hid_dim)
        self.fc         = nn.Linear((hid_dim * 2) + hid_dim + emb_dim, output_dim)
        self.dropout    = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs, mask):
        #input: [batch_size]
        #hidden: [batch_size, hid_dim]
        #encoder_ouputs: [src len, batch_size, hid_dim * 2]
        #mask: [batch_size, src len]
                
        #embed our input
        input    = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        #embedded = [1, batch_size, emb_dim]
        
        #calculate the attention
        a = self.attention(hidden, encoder_outputs, mask)
        #a = [batch_size, src len]
        a = a.unsqueeze(1)
        #a = [batch_size, 1, src len]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        #encoder_ouputs: [batch_size, src len, hid_dim * 2]
        weighted = torch.bmm(a, encoder_outputs)
        #weighted: [batch_size, 1, hid_dim * 2]
        weighted = weighted.permute(1, 0, 2)
        #weighted: [1, batch_size, hid_dim * 2]
        
        #send the input to decoder rnn
            #concatenate (embed, weighted encoder_outputs)
            #[1, batch_size, emb_dim]; [1, batch_size, hid_dim * 2]
        rnn_input = torch.cat((embedded, weighted), dim = 2)
        #rnn_input: [1, batch_size, emb_dim + hid_dim * 2]
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
            
        #send the output of the decoder rnn to fc layer to predict the word
            #prediction = fc(concatenate (output, weighted, embed))
        embedded = embedded.squeeze(0)
        output   = output.squeeze(0)
        weighted = weighted.squeeze(0)
        prediction = self.fc(torch.cat((embedded, output, weighted), dim = 1))
        #prediction: [batch_size, output_dim]
            
        return prediction, hidden.squeeze(0), a.squeeze(1)


####################################################################

from torchtext.vocab import build_vocab_from_iterator

def initialize_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def sequential_transforms(*transforms):
    # print(type(transforms[0]))
    def func(txt_input):
        for transform in transforms:
            # if transform.__class__.__name__ == 'module':
            #     print(txt_input)
            #     txt_input = transform(txt_input, lang='hi')
            # else:
            # # print("Hello", transform.__class__.__name__)
            txt_input = transform(txt_input)
        return txt_input
    return func


# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids):
    return torch.cat((torch.tensor([SOS_IDX]), 
                      torch.tensor(token_ids), 
                      torch.tensor([EOS_IDX])))


SRC_LANGUAGE = 'en'
TRG_LANGUAGE = 'ne'


# Define special symbols and indices
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<sos>', '<eos>']


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


input_dim   = 7333
output_dim  = 18612
emb_dim     = 256  
hid_dim     = 512  
dropout     = 0.5
SRC_PAD_IDX = 1

general_attn = Attention(hid_dim)
enc  = Encoder(input_dim,  emb_dim,  hid_dim, dropout)
dec1  = Decoder(output_dim, emb_dim,  hid_dim, dropout, general_attn)
model = Seq2SeqPackedAttention(enc, dec1, SRC_PAD_IDX, device).to(device)
model.apply(initialize_weights)

# model = torch.load('../models/best-val-lstm_lm.pt')


model.load_state_dict(torch.load('../models/Seq2SeqPackedAttentiongeneralatt.pt'))

en_vocab = torch.load('../models/en_vocab.pth')
ne_vocab = torch.load('../models/ne_vocab.pth')
vocab_transform = {'en':en_vocab, 'ne':ne_vocab}

vocab_transform[TRG_LANGUAGE](['here', 'is', 'a', 'unknownword', 'a'])


en_token = torch.load('../models/en_token.pth')
ne_token = torch.load('../models/ne_token.pth')
token_transform = {'en':en_token, 'ne':ne_token}

text_transform = {}
for ln in [SRC_LANGUAGE, TRG_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln], #Tokenization
                                               vocab_transform[ln], #Numericalization
                                               tensor_transform) # Add BOS/EOS and create tensor





def translate_eng_to_nepali(src):
    src_text = text_transform[SRC_LANGUAGE](src).to(device)
    src_text = src_text.reshape(-1, 1)
    trg_text = text_transform[TRG_LANGUAGE]('').to(device)
    trg_text = trg_text.reshape(-1, 1)
    text_length = torch.tensor([src_text.size(0)]).to(dtype=torch.int64)
    esc_chars = ['<sos>', '<unk>', '<eos>', '।']
    model.eval()
    with torch.no_grad():
        output, attentions = model(src_text, text_length, trg_text, 0) 
    output = output.squeeze(1)
    output = output[1:]
    output_max = output.argmax(1) 
    mapping = vocab_transform[TRG_LANGUAGE].get_itos()
    trg_tokens = ['<sos>'] + [mapping[token.item()] for token in output_max]
    trg_tokens = [trg for trg in trg_tokens if trg not in esc_chars]
    return ' '.join(trg_tokens)




# if __name__ == '__main__':
#     src = 'When he had made an end of prophesying, he came to the high place.'
#     print(translate_eng_to_nepali(src, model))