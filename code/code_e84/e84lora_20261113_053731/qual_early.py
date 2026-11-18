from common import *
p=C/'figures.py';s=p.read_text();s=s.replace(" fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)"," if os.environ.get('E84_QUAL_ONLY')=='1':return\n fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)")
s=s.replace("if __name__=='__main__':main();confusion_figure()","if __name__=='__main__':\n main()\n if os.environ.get('E84_QUAL_ONLY')!='1':confusion_figure()")
p.write_text(s)
print('qualitative_preview_enabled')
