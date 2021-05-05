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


#loads oct + vf significant values as well
class PROBDataLoader(torch.utils.data.Dataset):
    def __init__(self, csv_path, img_folder, subjective_csv = '/home/vip/sayan-mandal/datasets/obj_criteria/20200623-images_with_subjective.csv', \
         labeltype = 'mdprob',suspect =False,transform = None):
        df = pd.read_csv(csv_path, low_memory=False)
        df_subjective = pd.read_csv(subjective_csv)
        df_subjective = df_subjective[['maskedid']].drop_duplicates().reset_index(drop = True)
        df_subjective['with_subjective'] = 1

        df = df.join(df_subjective.set_index('maskedid'), on = 'maskedid')
        df = df.loc[df.with_subjective.isnull() == True].reset_index(drop = True)
        self.img_folder = img_folder
        self.transform = transform
        self.labeltype = labeltype    
        self.prob = {'Not Significant': 0,
                        '< 10%': 1,
                        '< 5%': 2,
                        '< 2%' : 3,
                        '< 1%' : 4,
                        '< 0.5%' : 5
                            }
        if not suspect:
            self.df = df.loc[df['classification2'] != 'suspect']
        else:
            self.df = df


    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        filename = self.df.iloc[index]["file_jpg"]
        label = self.prob[self.df.iloc[index][self.labeltype]]
        image = PIL.Image.open(os.path.join(self.img_folder,filename))
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class MDDataLoader(torch.utils.data.Dataset):
    def __init__(self, csv_path, img_folder, subjective_csv = '/home/vip/sayan-mandal/datasets/obj_criteria/20200623-images_with_subjective.csv', \
         labeltype = ['md_inf_prob','md_sup_prob'],suspect =False,transform = None):
        df = pd.read_csv(csv_path, low_memory=False)
        df_subjective = pd.read_csv(subjective_csv)
        df_subjective = df_subjective[['maskedid']].drop_duplicates().reset_index(drop = True)
        df_subjective['with_subjective'] = 1

        df = df.join(df_subjective.set_index('maskedid'), on = 'maskedid')
        df = df.loc[df.with_subjective.isnull() == True].reset_index(drop = True)
        self.img_folder = img_folder
        self.transform = transform
        self.labeltype = labeltype    
        self.prob = {'Not Significant': 0,
                        '< 10%': 1,
                        '< 5%': 2,
                        '< 2%' : 3,
                        '< 1%' : 4,
                        '< 0.5%' : 5
                            }
        if not suspect:
            self.df = df.loc[df['classification2'] != 'suspect']
        else:
            self.df = df


    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        filename = self.df.iloc[index]["file_jpg"]
        md_inf_prob = self.prob[self.df.iloc[index][self.labeltype[0]]]
        md_sup_prob = self.prob[self.df.iloc[index][self.labeltype[1]]]
        image = PIL.Image.open(os.path.join(self.img_folder,filename))
        if self.transform is not None:
            image = self.transform(image)
        return image, md_inf_prob, md_sup_prob


class RNFLDataLoader(torch.utils.data.Dataset):
    def __init__(self, csv_path, img_folder, subjective_csv = '/home/vip/sayan-mandal/datasets/obj_criteria/20200623-images_with_subjective.csv', \
         labeltype = 'rnflclass_g',suspect =False,transform = None):
        df = pd.read_csv(csv_path, low_memory=False)
        df_subjective = pd.read_csv(subjective_csv)
        df_subjective = df_subjective[['maskedid']].drop_duplicates().reset_index(drop = True)
        df_subjective['with_subjective'] = 1

        df = df.join(df_subjective.set_index('maskedid'), on = 'maskedid')
        df = df.loc[df.with_subjective.isnull() == True].reset_index(drop = True)
        self.img_folder = img_folder
        self.transform = transform
        self.labeltype = labeltype    
        self.prob = {'WNL': 0,
                    'BL': 1,
                    'ONL': 2
                    }
        if not suspect:
            self.df = df.loc[df['classification2'] != 'suspect']
        else:
            self.df = df


    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        filename = self.df.iloc[index]["file_jpg"]
        rnfl = self.prob[self.df.iloc[index][self.labeltype]]
        image = PIL.Image.open(os.path.join(self.img_folder,filename))
        if self.transform is not None:
            image = self.transform(image)
        return image, rnfl

    
class RNFLallDataLoader(torch.utils.data.Dataset):
    def __init__(self, csv_path, img_folder, subjective_csv = '/home/vip/sayan-mandal/datasets/obj_criteria/20200623-images_with_subjective.csv', \
         labeltype = ['rnflclass_ti','rnflclass_ni','rnflclass_ts','rnflclass_ns'],suspect =False,transform = None):
        df = pd.read_csv(csv_path, low_memory=False)
        df_subjective = pd.read_csv(subjective_csv)
        df_subjective = df_subjective[['maskedid']].drop_duplicates().reset_index(drop = True)
        df_subjective['with_subjective'] = 1

        df = df.join(df_subjective.set_index('maskedid'), on = 'maskedid')
        df = df.loc[df.with_subjective.isnull() == True].reset_index(drop = True)
        self.img_folder = img_folder
        self.transform = transform
        self.labeltype = labeltype    
        self.prob = {'WNL': 0,
                    'BL': 1,
                    'ONL': 2
                    }
        if not suspect:
            self.df = df.loc[df['classification2'] != 'suspect']
        else:
            self.df = df


    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        filename = self.df.iloc[index]["file_jpg"]
        rnflclass_ti = self.prob[self.df.iloc[index][self.labeltype[0]]]
        rnflclass_ni = self.prob[self.df.iloc[index][self.labeltype[1]]]
        rnflclass_ts = self.prob[self.df.iloc[index][self.labeltype[2]]]
        rnflclass_ns = self.prob[self.df.iloc[index][self.labeltype[3]]]
        image = PIL.Image.open(os.path.join(self.img_folder,filename))
        if self.transform is not None:
            image = self.transform(image)
        return image, rnflclass_ti,rnflclass_ni,rnflclass_ts,rnflclass_ns