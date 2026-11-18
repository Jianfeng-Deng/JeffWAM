from run import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.patches import Patch
analysis=json.loads((R/'worst_parent_analysis.json').read_text());colors=['#ffffff']+list(plt.get_cmap('tab10').colors)+['#333333'];cmap=ListedColormap(colors);norm=BoundaryNorm(np.arange(-1.5,11.5),cmap.N)
plots=[]
for a in analysis:
 parent=a['parent'];z=np.load(R/(parent+'_frozen_sequences.npz'));fig,axs=plt.subplots(4,1,figsize=(16,4.4),sharex=True);fig.subplots_adjust(left=.14,right=.99,top=.9,bottom=.21,hspace=.12)
 for ax,(name,y) in zip(axs,[('Reference (model)',z['truth'])]+[(k,z[k]) for k in ['S0','S1','S2']]):
  ax.imshow(y[None,:],aspect='auto',cmap=cmap,norm=norm,extent=[0,len(y),0,1],interpolation='nearest');ax.set_yticks([]);ax.set_ylabel(name,rotation=0,ha='right',va='center')
 axs[-1].set_xlabel('Original source frame (30 fps nominal)');fig.suptitle(parent+' | rank '+str(a['rank'])+' by frozen S2 accuracy')
 fig.legend(handles=[Patch(color=colors[i+1],label=s) for i,s in enumerate(S)]+[Patch(color='#333333',label='BATCH_WAIT')],loc='lower center',bbox_to_anchor=(.55,.035),ncol=11,frameon=False)
 path=R/(f"worst_{a['rank']}_"+parent+'_timeline_checked.png');assert not path.exists();fig.savefig(path,dpi=140);plt.close(fig);plots.append(str(path))

text=REPORT.read_text()
for a,p in zip(analysis,plots):text=text.replace(a['plot'].replace('.png','_legend.png'),p)
REPORT.write_text(text)
print('saved three plots with fixed margins')
