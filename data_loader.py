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

class FullDataLoader(torch.utils.data.Dataset):
    def __init__(self, csv_path, img_folder, subjective_csv = '/home/vip/sayan-mandal/datasets/obj_criteria/20200623-images_with_subjective.csv',suspect =False, transform = None):
        df = pd.read_csv(csv_path, low_memory=False)
        df_subjective = pd.read_csv(subjective_csv)
        df_subjective = df_subjective[['maskedid']].drop_duplicates().reset_index(drop = True)
        df_subjective['with_subjective'] = 1

        df = df.join(df_subjective.set_index('maskedid'), on = 'maskedid')
        df = df.loc[df.with_subjective.isnull() == True].reset_index(drop = True)
        self.img_folder = img_folder
        self.transform = transform
        self.classindex = {'suspect': 2,
                            'glaucoma':1, 
                            'normal' : 0}
        self.prob = {'Not Significant': 0,
                        '< 10%': 1,
                        '< 5%': 2,
                        '< 2%' : 3,
                        '< 1%' : 4,
                        '< 0.5%' : 5
                            }
        self.rnflp = {'WNL': 0,
                    'BL': 1,
                    'ONL': 2
                    }
        if not suspect:
            self.df = df.loc[df['classification2'] != 'suspect']
            self.classindex.pop('suspect')
        else:
            self.df = df


    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        filename = self.df.iloc[index]["file_jpg"]
        #glauc/msdprob/psdprob/mdinf/mdsup/rnflclass/rnflti/tnflni/rnflts/rnflns
        label1 = self.classindex[self.df.iloc[index]['classification2']]
        label2 = self.prob[self.df.iloc[index]['mdprob']]
        label3 = self.prob[self.df.iloc[index]['psdprob']]
        label4 = self.prob[self.df.iloc[index]['md_inf_prob']]
        label5 = self.prob[self.df.iloc[index]['md_sup_prob']]
        label6 = self.rnflp[self.df.iloc[index]['rnflclass_g']]
        label7 = self.rnflp[self.df.iloc[index]['rnflclass_ti']]
        label8 = self.rnflp[self.df.iloc[index]['rnflclass_ni']]
        label9 =self.rnflp[self.df.iloc[index]['rnflclass_ts']]
        label10 = self.rnflp[self.df.iloc[index]['rnflclass_ns']]
        image = PIL.Image.open(os.path.join(self.img_folder,filename))
        if self.transform is not None:
            image = self.transform(image)
        return image, label1,label2,label3,label4,label5,label6,label7,label8,label9,label10


class ProgressionDataLoader(torch.utils.data.Dataset):
    def __init__(self, csv_path, img_folder,suspect =True, transform = None):
        df = pd.read_csv(csv_path, low_memory=False)
        self.img_folder = img_folder
        self.transform = transform
        self.labeltype = 'class'   
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
        filename = self.df.iloc[index]["name"]
        label = self.classindex[self.df.iloc[index][self.labeltype]]
        time = self.df.iloc[index]["photodat_"]
        idx = self.df.iloc[index]["idx"]
        image = PIL.Image.open(os.path.join(self.img_folder,filename))
        if self.transform is not None:
            image = self.transform(image)
        return image, label, time, idx


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



class CSVFundusDataLoader(torch.utils.data.Dataset):
    def __init__(self, csv_path, img_folder, subjective_csv = '/home/vip/sayan-mandal/datasets/obj_criteria/20200623-images_with_subjective.csv', \
         labeltype = 'classification2',suspect =True, transform = None):
        df = pd.read_csv(csv_path, low_memory=False)
        #df_subjective = pd.read_csv(subjective_csv)
        #df_subjective = df_subjective[['maskedid']].drop_duplicates().reset_index(drop = True)
        #df_subjective['with_subjective'] = 1

        #df = df.join(df_subjective.set_index('maskedid'), on = 'maskedid')
        #df = df.loc[df.with_subjective.isnull() == True].reset_index(drop = True)
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
        return image, label,filename

class LSTMPrepDataLoader(torch.utils.data.Dataset):
    def __init__(self, csv_path, img_folder,transform = None):
        self.df = pd.read_csv(csv_path, low_memory=False)
        #df_subjective = pd.read_csv(subjective_csv)
        #df_subjective = df_subjective[['maskedid']].drop_duplicates().reset_index(drop = True)
        #df_subjective['with_subjective'] = 1

        #df = df.join(df_subjective.set_index('maskedid'), on = 'maskedid')
        #df = df.loc[df.with_subjective.isnull() == True].reset_index(drop = True)
        self.img_folder = img_folder
        self.transform = transform


    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        filename = self.df.iloc[index]["filename"]
        label = self.df.iloc[index]['label']
        maskedeye = self.df.iloc[index]['maskedeye']
        age = self.df.iloc[index]['age']
        pdays = self.df.iloc[index]['pdays']
        image = PIL.Image.open(os.path.join(self.img_folder,filename))
        if self.transform is not None:
            image = self.transform(image)
        return image, maskedeye, age, pdays, label