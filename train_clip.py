# code adapted from: https://github.com/openai/CLIP/issues/83 

import clip 
import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader 
from PIL import Image 

import pandas as pd 

EPOCH = 5 
BATCH_SIZE = 8 


# Latest Update : 18 July 2022, 09:55 GMT+7

# TO ADD :
# Gradient Checkpointing
# Filter out bias from weight decay
# Decaying learning rate with cosine schedule
# Half-precision Adam statistics
# Half-precision stochastically rounded text encoder weights were used

#BATCH_SIZE must larger than 1

device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
model, preprocess = clip.load("ViT-B/32",device=device,jit=False) #Must set jit=False for training

class image_title_dataset(Dataset):
    def __init__(self):
        self.vlmdata = pd.read_json('train_data/vlm.jsonl', lines=True) 
        self.length = 0 
        self.poss = [] 
        for i in range(len(self.vlmdata)): 
           self.poss.append( self.length ) 
           self.length += len(self.vlmdata.loc[i].annotations) 
        self.poss.append(self.length) 
        
        self.next_i = 0 

    def __len__(self):
        return self.length 

    def __getitem__(self, in_idx):
        idxset = False 
        if (self.poss[self.next_i] <= in_idx): 
            if (in_idx < self.poss[self.next_i+1]): 
                idx = self.next_i 
                idxset = True 
            elif (in_idx < self.poss[self.next_i+2]): 
               self.next_i += 1 
               idx = self.next_i 
               idxset = True 
        
        if (not idxset): 
            # binary search out idx 
            left=0 
            right=len(self.vlmdata)-1 
            
            while (right-left > 1): 
                mid = (left+right)//2 
                if (self.poss[mid] > in_idx): 
                    right = mid 
                else: 
                   left = mid 

            if (in_idx > self.poss[right]): 
               self.next_idx = right 
               idx = right 
            else: 
               self.next_idx = left 
               idx = left 


        image = preprocess(Image.open("train_data/images/"+self.vlmdata.loc[idx].image)) # Image from PIL module
        captions = [ self.vlmdata.loc[idx].annotations[i]['caption'] for i in range(len(self.vlmdata.loc[idx].annotations)) ]
        return image, captions 

# use your own data
dataset = image_title_dataset()
train_dataloader = DataLoader(dataset,batch_size = BATCH_SIZE) #Define your own dataloader

1/0 

#https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 


if device == "cpu":
  model.float()
else :
  clip.model.convert_weights(model) # Actually this line is unnecessary since clip by default already on float16

loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

# add your own code to track the training progress.
for epoch in range(EPOCH):
  for batch in train_dataloader :
      optimizer.zero_grad()

      images, texts = batch 
    
      images = images.to(device)
      texts = texts.to(device)
    
      logits_per_image, logits_per_text = model(images, texts)

      ground_truth = torch.arange(len(images),dtype=torch.long,device=device)

      total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
      total_loss.backward()
      if device == "cpu":
         optimizer.step()
      else : 
        convert_models_to_fp32(model)
        optimizer.step()
        clip.model.convert_weights(model)



# save model 

torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': total_loss,
        }, f"model_checkpoint/model_10.pt") #just change to your preferred folder/filename

