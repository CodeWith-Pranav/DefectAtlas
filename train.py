"""Offline MVTec bottle experiment. No test labels select checkpoints or thresholds."""
from pathlib import Path
import json,hashlib,base64,io,random,time,platform
import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision.models import mobilenet_v3_small,MobileNet_V3_Small_Weights
from sklearn.metrics import roc_auc_score,average_precision_score,confusion_matrix
ROOT=Path(__file__).resolve().parent
DATA=ROOT.parent/'defectatlas-data'/'bottle'
SEED=17;SIZE=96
class Autoencoder(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Conv2d(3,8,3,2,1),nn.ReLU(),nn.Conv2d(8,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.ConvTranspose2d(32,16,4,2,1),nn.ReLU(),nn.ConvTranspose2d(16,8,4,2,1),nn.ReLU(),nn.ConvTranspose2d(8,3,4,2,1),nn.Sigmoid())
 def forward(self,x):return self.net(x)
class Features(nn.Module):
 def __init__(self):
  super().__init__();self.net=mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1).features[:4]
  self.register_buffer('mean',torch.tensor([.485,.456,.406]).view(1,3,1,1));self.register_buffer('std',torch.tensor([.229,.224,.225]).view(1,3,1,1))
 def forward(self,x):return self.net((x-self.mean)/self.std)
def load(p):return torch.tensor(np.asarray(Image.open(p).convert('RGB').resize((SIZE,SIZE),Image.Resampling.NEAREST)).copy().transpose(2,0,1),dtype=torch.float32)/255

def main():
 torch.set_num_threads(2);torch.manual_seed(SEED);random.seed(SEED);np.random.seed(SEED)
 paths=sorted((DATA/'train/good').glob('*.png'));assert len(paths)==209, 'Complete bottle training set (209 images) required';perm=np.random.default_rng(SEED).permutation(len(paths));cut=int(.8*len(paths));tr=[paths[i] for i in perm[:cut]];va=[paths[i] for i in perm[cut:]]
 train=torch.stack([load(p) for p in tr]);val=torch.stack([load(p) for p in va]);ae=Autoencoder();opt=torch.optim.Adam(ae.parameters(),lr=.002);best=float('inf');history=[]
 for epoch in range(35):
  ae.train();losses=[]
  for ids in torch.randperm(len(train)).split(24):
   x=train[ids];noisy=(x+torch.randn_like(x)*.02).clamp(0,1);loss=((ae(noisy)-x)**2).mean();opt.zero_grad();loss.backward();opt.step();losses.append(loss.item())
  ae.eval()
  with torch.inference_mode():vl=((ae(val)-val)**2).mean().item()
  history.append({'epoch':epoch+1,'trainMSE':float(np.mean(losses)),'validationMSE':vl})
  if vl<best:best=vl;torch.save(ae.state_dict(),ROOT/'autoencoder.pt');best_epoch=epoch+1
  if (epoch+1)%5==0:print('epoch',epoch+1,'val',round(vl,6),flush=True)
 ae.load_state_dict(torch.load(ROOT/'autoencoder.pt',weights_only=True));ae.eval();features=Features().eval()
 with torch.inference_mode():normal_features=features(train);mu=normal_features.mean(0);std=normal_features.std(0,unbiased=False).clamp(min=.05)
 def maps(x):
  with torch.inference_mode():
   reconstruction=ae(x);f=features(x);a=nn.functional.avg_pool2d(((reconstruction-x)**2).mean(1,keepdim=True),8)[:,0];b=(((f-mu)/std)**2).mean(1)
   return {'ae-v1':a.numpy(),'feature-v1':b.numpy()}
 def score(m):return np.sort(m.reshape(len(m),-1),axis=1)[:,-8:].mean(1)
 vm=maps(val);thresholds={k:float(np.quantile(score(v),.95)) for k,v in vm.items()};pixel_scales={k:float(np.quantile(v,.99)) for k,v in vm.items()}
 test_paths=sorted((DATA/'test').glob('*/*.png'));assert len(test_paths)==83, 'Complete bottle test set (83 images) required';test_x=torch.stack([load(p) for p in test_paths]);labels=np.array([int(p.parent.name!='good') for p in test_paths]);tm=maps(test_x)
 report={};test_rows=[]
 for k,v in tm.items():
  scores=score(v);threshold=thresholds[k];tn,fp,fn,tp=confusion_matrix(labels,scores>=threshold,labels=[0,1]).ravel()
  per_type={t:{'n':sum(p.parent.name==t for p in test_paths),'flagged':int(sum(p.parent.name==t and s>=threshold for p,s in zip(test_paths,scores)))} for t in sorted(set(p.parent.name for p in test_paths))}
  report[k]={'name':'Denoising convolutional autoencoder' if k=='ae-v1' else 'Pretrained MobileNet spatial baseline','threshold':threshold,'pixelScale':pixel_scales[k],'auroc':float(roc_auc_score(labels,scores)),'averagePrecision':float(average_precision_score(labels,scores)),'tp':int(tp),'tn':int(tn),'fp':int(fp),'fn':int(fn),'recall':float(tp/(tp+fn)),'precision':float(tp/(tp+fp)) if tp+fp else 0,'falsePositiveRate':float(fp/(fp+tn)),'perType':per_type,'normalValidationExceedances':int(sum(score(vm[k])>=threshold)),'parameters':sum(p.numel() for p in (ae if k=='ae-v1' else features).parameters())}
 for i,p in enumerate(test_paths):test_rows.append({'path':str(p.relative_to(DATA)),'label':int(labels[i]),'defect':p.parent.name,'scores':{k:float(score(v)[i]) for k,v in tm.items()}})
 # Keep a deterministic, labeled gallery (first two per test type); this is not a separate validation set.
 gallery=[]
 for t in sorted(set(p.parent.name for p in test_paths)):
  for p in [p for p in test_paths if p.parent.name==t][:2]:
   im=Image.open(p).convert('RGB').resize((96,96),Image.Resampling.NEAREST);b=io.BytesIO();im.save(b,format='PNG')
   gallery.append({'id':t+'-'+p.stem,'label':t,'actual':int(t!='good'),'source':str(p.relative_to(DATA)),'image':'data:image/png;base64,'+base64.b64encode(b.getvalue()).decode()})
 torch.onnx.export(ae,torch.zeros(1,3,SIZE,SIZE),str(ROOT/'autoencoder.onnx'),input_names=['image'],output_names=['reconstruction'],opset_version=17,dynamo=False)
 torch.onnx.export(features,torch.zeros(1,3,SIZE,SIZE),str(ROOT/'features.onnx'),input_names=['image'],output_names=['features'],opset_version=17,dynamo=False)
 golden=[]
 for i in [0,len(test_paths)//2,len(test_paths)-1]:
  golden.append({'path':str(test_paths[i].relative_to(DATA)),'tensor':base64.b64encode(test_x[i].numpy().astype('<f4').tobytes()).decode(),'scores':{k:float(score(v)[i]) for k,v in tm.items()}})
 bundle={'version':'1.0.0','category':'bottle','inputSize':SIZE,'mapSize':12,'topPixels':8,'seed':SEED,'modelPolicy':'feature-v1','models':report,'featureMean':mu.numpy().reshape(-1).tolist(),'featureStd':std.numpy().reshape(-1).tolist(),'featureShape':list(mu.shape),'history':history,'bestEpoch':best_epoch,'dataset':{'name':'MVTec AD / bottle','synthetic':False,'license':'CC BY-NC-SA 4.0','source':'https://www.mvtec.com/research-teaching/datasets/mvtec-ad','mirror':'https://huggingface.co/datasets/aditiparashar/mvtec-ad','training':len(tr),'validation':len(va),'test':len(test_paths),'testDefects':int(labels.sum()),'testHealthy':int((1-labels).sum()),'trainFiles':[str(p.relative_to(DATA)) for p in tr],'validationFiles':[str(p.relative_to(DATA)) for p in va]},'gallery':gallery,'testRows':test_rows,'runtime':{'torch':torch.__version__,'python':platform.python_version(),'numpy':np.__version__},'pretrained':'MobileNet_V3_Small_Weights.IMAGENET1K_V1; first 4 feature blocks; fixed ImageNet weights','thresholdPolicy':'95th percentile of held-out healthy validation image scores; no test labels used','scorePolicy':'mean of largest 8 pixels on a 12×12 anomaly map'}
 (ROOT/'model-card.json').write_text(json.dumps(bundle,separators=(',',':'),allow_nan=False));(ROOT/'golden.json').write_text(json.dumps(golden,separators=(',',':')))
 print(json.dumps({'bestEpoch':best_epoch,'models':report,'dataset':{k:bundle['dataset'][k] for k in ['training','validation','test','testDefects','testHealthy']}},indent=2),flush=True)
if __name__=='__main__':main()
