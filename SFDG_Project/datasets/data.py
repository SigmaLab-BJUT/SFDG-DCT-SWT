import os 
import json
import random
dfdcroot='/mnt/disk2/wy/dfdcimage/'
dfdc_test_toot = '/mnt/disk2/wy/DFDC/'
celebroot='/mnt/disk2/wy/Celeb-DF/'
ffpproot='/mnt/disk1/wy/FF++/'
deeperforensics_root='datasets/deeper/'
wild_deepfake_root = '/mnt/disk2/wy/WildDeepfake/'

def load_json(name):
    with open(name) as f:
        a=json.load(f)
    return a


def catdir(dir,label):
    l=os.listdir(dir)
    return [[os.path.join(dir,i),label] for i in l]


def FF_dataset(tag='original_sequences/youtube',codec='c23',part='train'):
    assert(tag in ['original_sequences/youtube','manipulated_sequences/Deepfakes','manipulated_sequences/NeuralTextures',
                    'manipulated_sequences/FaceSwap','manipulated_sequences/Face2Face'])
    assert(codec in ['c0','c23','c40','all'])
    assert(part in ['train','valid','test','all'])
    if part=="all":
        return FF_dataset(tag,codec,'train')+FF_dataset(tag,codec,'valid')+FF_dataset(tag,codec,'test')
    if codec=='all':
        return FF_dataset(tag,'c0',part)+FF_dataset(tag,'c23',part)+FF_dataset(tag,'c40',part)
    # if tag == 'original_sequences/youtube' and  part == 'test':
    #     codec = 'c23'

    path=ffpproot+'%s/%s/images/'%(tag,codec)
    metafile=load_json(ffpproot+part+'.json') 
    files=[]
    if tag=='original_sequences/youtube':
        for i in metafile:
            files.append([path+i[0],0])
            files.append([path+i[1],0])
        
    else:
        for i in metafile:
            files.append([path+i[0]+'_'+i[1],1])
            files.append([path+i[1]+'_'+i[0],1])

    return files

Celeb_test=list(map(lambda x:[os.path.join(celebroot,x[0]),1-x[1]],load_json(celebroot+'celeb.json')))

WildDeepfake_test = list(map(lambda x:[x[0],x[1]],load_json(wild_deepfake_root+'WildDeepfake_test.json')))[200:600]
WildDeepfake_train = list(map(lambda x:[x[0],x[1]],load_json(wild_deepfake_root+'WildDeepfake_train.json')))
# print(len(WildDeepfake_train))

DFDC_train = list(map(lambda x:[x[0],x[1]],load_json(dfdcroot+'DFDC_train.json')))

def make_balance(data):
    tr=list(filter(lambda x:x[1]==0,data))
    tf=list(filter(lambda x:x[1]==1,data))
    if len(tr)>len(tf):
        tr,tf=tf,tr
    rate=len(tf)//len(tr)
    res=len(tf)-rate*len(tr)
    tr=tr*rate+random.sample(tr,res)
    return tr+tf

def dfdc_dataset(part='train'):
    assert (part in ['train', 'valid', 'test'])
    if part == 'train':
        files = list(map(lambda x:[x[0],x[1]],load_json(dfdcroot+'DFDC_train.json')))
    else:
        lf = load_json(dfdc_test_toot + 'DFDC.json')
        path = dfdc_test_toot + 'train_sample_frames/'
        files = [[path + i[0], i[1]] for i in lf]
    # if part == 'train':
    #     path = dfdcroot + 'dfdc/'
    #     files = make_balance(lf['train'])
    # if part == 'test':
    #     path = dfdcroot + 'dfdc-test/'
    #     files = lf['test']
    # if part == 'val':
    #     path = dfdcroot + 'dfdc-val/'
    #     files = lf['val']
    # path = os.path.join(path, 'train_sample_frames/')
    # path = dfdcroot + 'train_sample_frames/'
    
    return files
    
def deeperforensics_dataset(part='train'):
    a=os.listdir(deeperforensics_root)
    d=dict()
    for i in a:
        d[i.split('_')[0]]=i
    l=lambda x:[deeperforensics_root+d[x],1]
    metafile=load_json(ffpproot+part+'.json')
    files=[]
    for i in metafile:
        files.append(l(i[0]))
        files.append(l(i[1]))
    return files
    