import nltk
import os
import torch
import torch.utils.data as data
from vocabulary import Vocabulary
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
from tqdm import tqdm
import random
import json





def get_val_loader(transform,
               batch_size=1,
               vocab_file='./vocab.pkl',
               start_word="<start>",
               end_word="<end>",
               unk_word="<unk>",
               vocab_from_file=True,
               num_workers=0,
               cocoapi_loc='opt'):

   
    assert os.path.exists(vocab_file), "Must first generate vocab.pkl from training data."
    assert vocab_from_file==True, "Change vocab_from_file to True."
    img_folder = os.path.join(cocoapi_loc, 'cocoapi/images/val2014/')
    annotations_file = os.path.join(cocoapi_loc, 'cocoapi/annotations/captions_val2014.json')

    # COCO caption dataset.
    dataset = CoCoDataset(transform=transform,
                          batch_size=batch_size,
                          vocab_threshold=None,
                          vocab_file=vocab_file,
                          start_word=start_word,
                          end_word=end_word,
                          unk_word=unk_word,
                          annotations_file=annotations_file,
                          vocab_from_file=vocab_from_file,
                          img_folder=img_folder)


 
    data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      num_workers=num_workers,drop_last=False)

    return data_loader

class CoCoDataset(data.Dataset):
    
    def __init__(self, transform, batch_size, vocab_threshold, vocab_file, start_word, 
        end_word, unk_word, annotations_file, vocab_from_file, img_folder):
        self.transform = transform
        self.batch_size = batch_size
        self.vocab = Vocabulary(vocab_threshold, vocab_file, start_word,
            end_word, unk_word, annotations_file, vocab_from_file)
        self.img_folder = img_folder
        
        self.coco = COCO(annotations_file)
        self.ids = list(self.coco.anns.keys())
        self.img_ids=list(self.coco.imgs.keys()) 
        print('Obtaining caption lengths...')
        print(len(self.img_ids))
        all_tokens = [nltk.tokenize.word_tokenize(str(self.coco.anns[self.ids[index]]['caption']).lower()) 
                      for index in tqdm(np.arange(len(self.ids)))]
        self.caption_lengths = [len(token) for token in all_tokens]
        
    def __getitem__(self, index):
        # obtain image and caption if in training mode
        
        img_id=self.img_ids[index]
        path = self.coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.img_folder, path)).convert('RGB')
        image = self.transform(image)

        return img_id,image


    def __len__(self):
       
       return len(self.img_ids)
    
    
    
class HoHoDataset(data.Dataset):
    
    def __init__(self, transform, batch_size, vocab_threshold, vocab_file, start_word, 
        end_word, unk_word, annotations_file, vocab_from_file, img_folder):
        self.transform = transform
        self.batch_size = batch_size
        self.vocab = Vocabulary(vocab_threshold, vocab_file, start_word,
            end_word, unk_word, annotations_file, vocab_from_file)
        self.img_folder = img_folder
        
        self.coco = COCO(annotations_file)
        self.ids = list(self.coco.anns.keys())
        self.img_ids=list(self.cooc.imgs.keys()) 
        print('Obtaining caption lengths...')
        print(len(img_ids))
        all_tokens = [nltk.tokenize.word_tokenize(str(self.coco.anns[self.ids[index]]['caption']).lower()) for index in tqdm(np.arange(len(self.ids)))]
        self.caption_lengths = [len(token) for token in all_tokens]
        
    def __getitem__(self, index):
        # obtain image and caption if in training mode
        ann_id = self.ids[index]
        caption = self.coco.anns[ann_id]['caption']
        img_id = self.coco.anns[ann_id]['image_id']
        path = self.coco.loadImgs(img_id)[0]['file_name']

        # Convert image to tensor and pre-process using transform
        #print(path)
        image = Image.open(os.path.join(self.img_folder, path)).convert('RGB')
        image = self.transform(image)

        # Convert caption to tensor of word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(self.vocab(self.vocab.start_word))
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab(self.vocab.end_word))
        #caption = torch.Tensor(caption).long()

        # return pre-processed image and caption tensors
       # return (img_id,image, caption)
        return img_id,image,caption

       
    def get_train_indices(self):
        sel_length = np.random.choice(self.caption_lengths)
        all_indices = np.where([self.caption_lengths[i] == sel_length for i in np.arange(len(self.caption_lengths))])[0]
        indices = list(np.random.choice(all_indices, size=self.batch_size))
        return indices

    def __len__(self):
       
       return len(self.ids)
       
       