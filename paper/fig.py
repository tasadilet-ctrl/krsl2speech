import json, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Times','DejaVu Serif'],'font.size':8.5,
  'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#888','axes.linewidth':0.6,
  'xtick.color':'#444','ytick.color':'#444','xtick.major.width':0.6,'ytick.major.width':0.6})
r=json.load(open('paper_runs.json'))
fig,ax=plt.subplots(figsize=(3.15,2.45))
ax.grid(True,color='#e6e6e6',lw=0.5); ax.set_axisbelow(True)
ax.axvline(26,color='#999',lw=0.8,ls=(0,(3,2)))
ax.text(26.4,3.2,'takeoff rule\n(chrF$\\geq$26, recall$\\geq$0.08)',fontsize=6.5,color='#555',va='top')
for grp,col,mk,lab in [(False,'#2a78d6','o','original references (36 runs)'),(True,'#eb6834','^','rewritten references, E16 (6 runs)')]:
    xs=[x for x in r if (x['arm']=='E16')==grp]
    ax.errorbar([x['chrf'] for x in xs],[x['coup'] for x in xs],
        yerr=[[x['coup']-x['lo'] for x in xs],[x['hi']-x['coup'] for x in xs]],fmt='none',ecolor=col,elinewidth=0.6,alpha=0.6)
    ax.scatter([x['chrf'] for x in xs],[x['coup'] for x in xs],s=26 if grp else 20,marker=mk,c=col,edgecolors='white',linewidths=0.6,label=lab,zorder=3)
ax.set_xlabel('chrF, dev selection set (600 clips)'); ax.set_ylabel('coupling $\\Delta$ (chrF points)')
ax.set_xlim(10,37); ax.set_ylim(-0.5,15.5)
ax.legend(loc='upper left',fontsize=6.5,frameon=False,handletextpad=0.3,borderaxespad=0.2)
fig.tight_layout(pad=0.3); fig.savefig('fig_runs.pdf'); fig.savefig('fig_runs.png',dpi=200)
