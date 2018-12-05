import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F





class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        #resnet = models.resnet50(pretrained=True)
        resnet = models.resnet152(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.batchnorm = nn.BatchNorm1d(embed_size, momentum=0.01)


    def forward(self, images):
        
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        features = self.batchnorm(features)
        return features     

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size,num_layers=1,bidirectional=True, dropout_p=0.1):
        super(DecoderRNN, self).__init__()
        self.hidden_size=hidden_size
        self.embedding=nn.Embedding(vocab_size,embed_size)
        self.lstm=nn.LSTM(embed_size,hidden_size,num_layers, batch_first=True,bidirectional=bidirectional,dropout=dropout_p)
       # self.dropout=nn.Dropout(dropout_p)
        if bidirectional:
            self.output=nn.Linear(hidden_size*2,vocab_size)
        else:
            self.output=nn.Linear(hidden_size,vocab_size)
            
    def forward(self, features, captions):
        out=captions[:,:-1]
        out=self.embedding(out)
        out=torch.cat((features.unsqueeze(1),out),1)
        out,s= self.lstm(out)
       # print(out.shape)
        out= self.output(out)
        return out
    
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        sample_id=[]
        #inputs=inputs.unsqueeze(1)
        for i in range(max_len):
            hidden,states=self.lstm(inputs,states)
            out=self.output(hidden.squeeze(1))
            predicted=out.argmax(1)
            sample_id.append(predicted.item())

            inputs=self.embedding(predicted)
            inputs=inputs.unsqueeze(1)
        return sample_id 
    
    #def beam_search(self,sentences,max_len=20,top_pred=3):
    def beam_search(self,inputs,states=None,max_len=20,top_pred=3):
        sentences=[]
        sentences.append([[],0,inputs,states])
        
        for i in range(max_len):
            sample_id = []
            for sentence,score,inputs,states in sentences:
                hidden, states = self.lstm(inputs, states)
                out= self.output(hidden.squeeze(1))

                scores = F.log_softmax(out, -1)
                
                new_score, new_idx = scores.topk(top_pred, 1)# ,True,True)
                
                new_idx = new_idx.squeeze(0)
                
                for i in range(top_pred):
                    new_sentence,sentence_scores=sentence.copy(),score
                    
                    new_sentence.append(new_idx[i].item())
                    sentence_scores += new_score[0][i].item()

                    inputs = self.embedding(new_idx[i].unsqueeze(0)).unsqueeze(0)
                    sample_id.append([new_sentence, sentence_scores, inputs, states])
            
            sample_id.sort(key=lambda x:x[1],reverse=True)
            sentences=sample_id[:top_pred]
        return [sentence[0] for sentence in sentences]    
    
    
    
class EEncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EEncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        #resnet = models.resnet152(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        
        

    def forward(self, images):
        
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        #features = self.batchnorm(features)
        return features  
    
class DecoderRNN2(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN2, self).__init__()
        self.hidden_size=hidden_size
        self.embedding=nn.Embedding(vocab_size,embed_size)
        self.lstm=nn.LSTM(embed_size,hidden_size,num_layers, batch_first=True)
        self.dropout=nn.Dropout(0.2)
        self.output=nn.Linear(hidden_size,vocab_size)
    
    def forward(self, features, captions):
        #print(captions.shape)
        out=captions[:,:-1]
        #print(out.shape,'removed last word')
        out=self.embedding(out)
        #print(out.shape,'embed')
        out=torch.cat((features.unsqueeze(1),out),1)
        #print(out.shape,'concat')
        out,s= self.lstm(out)
        out= self.output(out)
        return out
    
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        sample_id=[]
        #inputs=inputs.unsqueeze(1)
        for i in range(max_len):
            hidden,states=self.lstm(inputs,states)
            out=self.output(hidden.squeeze(1))
            predicted=out.argmax(1)
            sample_id.append(predicted.item())

            inputs=self.embedding(predicted)
            inputs=inputs.unsqueeze(1)
        return sample_id 
    

class AttnDecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size,dropout_p=0.1, max_length=10):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.attn = nn.Linear(self.embed_size, self.hidden_size)
        self.attn_combine = nn.Linear(self.hidden_size, self.hidden_size * 2)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size * 2, self.hidden_size * 2,batch_first=True)
        self.out = nn.Linear(self.hidden_size * 2, self.vocab_size)

    def forward(self, features, captions, encoder_outputs):
        
        embed=captions[:,:-1]
        embed=self.embedding(embed)
        embed = self.dropout(embed)
        
        attn_weights=F.softmax(
            self.attn(torch.cat((features.unsqueeze(1),embed),1)),dim=1)
        print(attn_weights.shape)
        print(features.shape)
        print(features.unsqueeze(0).shape)
        attn_applied = torch.bmm(attn_weights,
                                 #encoder_outputs.unsqueeze(0))
                                 features.unsqueeze(0))

        output = torch.cat((embed, attn_applied), 1)
        output = self.attn_combine(output).unsqueeze(0)

        #output = F.relu(output)
        output, h = self.gru(output)
        output= self.output(output)
        return output, h, attn_weights

    
    

        
        
        
       
    
    
    
    
    