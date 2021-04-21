import pandas as pd
import torch
from torchvision import transforms
import PIL
import os

class FundusDataLoader(torch.utils.data.Dataset):
    def __init__(self, csv_path, img_folder, subjective_csv = '/home/vip/sayan-mandal/datasets/obj_criteria/20200623-images_with_subjective.csv', \
         labeltype = 'classification2',suspect =False, transform = None):
        df = pd.read_csv(csv_path, low_memory=False)
        df_subjective = pd.read_csv(subjective_csv)
        df_subjective = df_subjective[['maskedid']].drop_duplicates().reset_index(drop = True)
        df_subjective['with_subjective'] = 1

        df = df.join(df_subjective.set_index('maskedid'), on = 'maskedid')
        df = df.loc[df.with_subjective.isnull() == True].reset_index(drop = True)
        self.img_folder = img_folder
        self.transform = transform
        self.labeltype = labeltype    
        self.classindex = {'suspect': 2,
                            'glaucoma':1, 
                            'normal' : 0}
        if not suspect:
            self.df = df.loc[df[labeltype] != 'suspect']
            self.classindex.pop('suspect')
        else:
            self.df = df


    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        filename = self.df.iloc[index]["file_jpg"]
        label = self.classindex[self.df.iloc[index][self.labeltype]]
        image = PIL.Image.open(os.path.join(self.img_folder,filename))
        if self.transform is not None:
            image = self.transform(image)
        return image, label

