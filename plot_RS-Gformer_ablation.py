# -*- coding: utf-8 -*-
"""PRG-Gformer ablation: same typography, borders and grid as the user's line plots.
Run: python plot_PRG_ablation_styleA.py
Results are plotted exactly as specified in the rates dictionary.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MultipleLocator

K = 4; NT = 8; mu = 0.2

rates = {
    'w/o Robust': 5.680965,
    'w/o RiskGate': 5.446037,
    'SecGNN Robust': 3.773253,
    'Full PRG': 6.240090,
}

metric_mode = 'relative'
save_dir = './graphs/exp_ablation'

font_list = {f.name for f in font_manager.fontManager.ttflist}
font_name = 'Times New Roman' if 'Times New Roman' in font_list else 'Liberation Serif'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = [font_name, 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

labels = list(rates)
colors = ['#AD8AE8','#0072BD','#E5AB36','#D95319']
hatches = ['////','xxx','---','']
xticklabels = ['w/o Robust','w/o RiskGate','SecGNN Robust','Full PRG']

def plot_ablation(figure_mode='wide'):
    if rates['Full PRG'] is None or rates['Full PRG'] <= 0:
        raise ValueError('Full PRG must have a positive measured worst-case SSR.')
    if metric_mode not in ('relative','rate'):
        raise ValueError("metric_mode must be 'relative' or 'rate'")
    wide = figure_mode == 'wide'
    if figure_mode not in ('wide','ieee'):
        raise ValueError("figure_mode must be 'wide' or 'ieee'")
    fig,ax = plt.subplots(figsize=(7.0,4.8) if wide else (6.0,4.4),dpi=150)
    fs = 16
    xtfs = 16
    valfs = 16
    width = 0.59 if wide else 0.63
    xs = np.arange(len(labels))
    full_rate = float(rates['Full PRG'])
    measured = [float(v) for v in rates.values() if v is not None]
    ymax = 115 if metric_mode == 'relative' else max(measured)*1.22
    for i,(name,rate) in enumerate(rates.items()):
        if rate is None:
            ax.text(xs[i],ymax*.51,'TBD',fontsize=valfs,color='#666666',
                    ha='center',va='center',fontstyle='italic')
            continue
        y = (100*rate/full_rate) if metric_mode == 'relative' else float(rate)
        ax.bar(xs[i],y,width=width,color=colors[i],edgecolor='#333333',
               linewidth=1.15 if wide else .85,hatch=hatches[i],zorder=3)
        vlabel = f'{y:.2f}%' if metric_mode == 'relative' else f'{y:.2f}'
        ax.annotate(vlabel,(xs[i],y),xytext=(0,6 if wide else 4),
                    textcoords='offset points',ha='center',va='bottom',
                    fontsize=valfs,color='#262626',fontweight='medium')
    ax.set_xlim(-0.54,3.54)
    ax.set_ylim(0,ymax)
    ax.set_xticks(xs)
    ax.set_xticklabels(xticklabels,fontsize=xtfs)
    ax.set_ylabel('Worst-case SSR / Full PRG (%)' if metric_mode=='relative'
                  else 'Worst-case Sum Secrecy Rate (bps/Hz)',fontsize=18)
    ax.yaxis.set_major_locator(MultipleLocator(20 if metric_mode=='relative' else 2))
    ax.tick_params(axis='both',labelsize=fs ,direction='in', length=5 ,width=.9)
    ax.tick_params(axis='x',labelsize=14,pad=7 if wide else 5)
    ax.grid(True,axis='both',linestyle='-',linewidth=0.6,
            color='#BFBFBF',alpha=0.45,zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    for spine in ax.spines.values(): spine.set_linewidth(.9 if wide else .8)

    fig.tight_layout(pad=.8 if wide else .5)
    os.makedirs(save_dir,exist_ok=True)
    name = f'PRG_Ablation_{metric_mode}_{figure_mode}_K{K}_NT{NT}_mu{mu}'
    for ext in ('png','pdf'):
        fig.savefig(os.path.join(save_dir,f'{name}.{ext}'),
                    dpi=400,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    print('Saved:',os.path.join(save_dir,name))

if __name__=='__main__':
    plot_ablation('wide')
    plot_ablation('ieee')
    print('Known: Full PRG = 100%, w/o Robust =',
          f"{100*rates['w/o Robust']/rates['Full PRG']:.2f}%")
