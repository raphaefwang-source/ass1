"""Figures are derived only from saved CSVs, in the original scientific style."""
import experiment as X
from experiment import np
import paper_figures as P
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

def save(fig,name):
    for ext in ('png','svg'):fig.savefig(X.FIGURES/f'{name}.{ext}',dpi=240,bbox_inches='tight',pad_inches=.05)
    plt.close(fig)

PANELS=(('sR','(a) Real-space cutoff',r'$\xi r_c$',False),
        ('sT','(b) Positive-table image window',r'$\kappa R_T$ (cube half-width)',False),
        ('M','(c) FFT mesh size',r'$M$',True),
        ('p','(d) B-spline assignment order',r'$p$',False),
        ('rho','(e) Particle density',r'$\rho$',False),
        ('kappa','(f) Screening parameter',r'$\kappa$',True))

def six(rows,metric):
    fig,axs=plt.subplots(2,3,figsize=(8.4,5.5),sharey=True)
    series=[]
    for ax,(sw,title,xl,logx) in zip(axs.ravel(),PANELS):
        for st in ('baseline','moderate'):
            for fam in P.FAMS:
                rs=[r for r in rows if r['sweep']==sw and r['setting']==st and r['family']==fam]
                if not rs:continue
                xx,yy,lo,hi=P.stats(rs,'value',metric)
                P.curve(ax,xx,yy,lo,hi,P.FAM_COL[fam],P.FAM_MK[fam],st=='moderate',ms=P.A_FAM_MS[fam])
                series += [dict(metric=metric,sweep=sw,setting=st,family=fam,x=a,median=b,min=c,max=d)
                           for a,b,c,d in zip(xx,yy,lo,hi)]
        ax.set_yscale('log');ax.set_ylim(1e-5,1.)
        if logx:ax.set_xscale('log')
        if sw in ('sR','sT'):ax.text(.97,.97,'baseline only',transform=ax.transAxes,ha='right',va='top',fontsize=6.5)
        if sw=='M':
            ax.set_xticks([16,24,32,48,64,80]);ax.set_xticklabels(['16','24','32','48','64','80'])
            ax.minorticks_off()
        if sw=='p':ax.set_xticks(range(3,10))
        if sw=='kappa':
            ax.set_xticks([.2,.5,.7,1,2]);ax.set_xticklabels(['0.2','0.5','0.7','1','2']);ax.minorticks_off()
        ax.set_title(title);ax.set_xlabel(xl)
    for ax in axs[:,0]:ax.set_ylabel(r'$E_\Gamma$ (relative spectral norm)' if metric=='E_G' else r'$E_{\Gamma X}$ (10 Gaussian probes)')
    handles=[Line2D([],[],color=P.FAM_COL[f],marker=P.FAM_MK[f],label=P.A_FAM_LAB[f]) for f in P.FAMS]
    handles += [Line2D([],[],color='k',label=r'baseline: $s_R=6$, $\kappa R_T=36$, $M=80$, $p=9$'),
                Line2D([],[],color='k',ls='--',label=r'moderate: $s_R=4$, $\kappa R_T=24$, $M=40$, $p=7$')]
    fig.legend(handles=handles,loc='lower center',ncol=3,fontsize=6.6)
    fig.tight_layout(rect=(0,.09,1,1),h_pad=1.2)
    save(fig,'fig1_operator_six_sweeps' if metric=='E_G' else 'fig2_action_six_sweeps')
    return series

def comparison(rows):
    old=X.read(X.HERE.parent/'accuracy_sweeps/results/sweeps.csv')
    fig,axs=plt.subplots(1,2,figsize=(7.7,3.1),sharey=True)
    for ax,sw in zip(axs,('M','p')):
        for fam in P.FAMS:
            for data,ls,label in ((rows,'-',f'{fam}: B2'),(old,':',f'{fam}: original PPPM')):
                rs=[r for r in data if r['sweep']==sw and r['setting']=='baseline' and r['family']==fam]
                xx,yy,lo,hi=P.stats(rs,'value','E_G')
                ax.plot(xx,yy,ls,color=P.FAM_COL[fam],marker=P.FAM_MK[fam],ms=3,label=label)
                ax.fill_between(xx,lo,hi,color=P.FAM_COL[fam],alpha=.08)
        ax.set_yscale('log');ax.set_xlabel('$M$' if sw=='M' else '$p$')
        ax.set_title('Mesh convergence' if sw=='M' else 'Increasing assignment order')
        if sw=='M':ax.set_xscale('log')
        else:ax.set_xticks(range(3,10))
    axs[0].set_ylabel(r'$E_\Gamma$ against the same $\Gamma_D$')
    axs[0].legend(fontsize=5.7,ncol=2)
    fig.tight_layout();save(fig,'fig3_original_pppm_comparison')

def optimization():
    search=X.read(X.RESULTS/'search.csv');timing=X.read(X.RESULTS/'final_timing.csv')
    if not timing:return
    fig,axs=plt.subplots(1,3,figsize=(9.1,3.))
    ax=axs[0]
    for p in sorted({r['p'] for r in search}):
        rs=[r for r in search if r['p']==p]
        ax.scatter([r['t_screen_s']*1000 for r in rs],[r['E_G'] for r in rs],s=10,alpha=.5,label=f'p={p}')
    ax.axhline(X.TOL,color='k',ls=':');ax.set_xscale('log');ax.set_yscale('log');ax.legend()
    ax.set_xticks([40,100,300,600]);ax.set_xticklabels(['40','100','300','600'])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel('screening matvec (ms; guide only)');ax.set_ylabel(r'$E_\Gamma$, screening configuration')
    ax.set_title('(a) Measured constrained search')
    rs=sorted(timing,key=lambda r:r['median_matvec_s'])
    b2=[r for r in rs if r['method']=='B2'];best=b2[0];old=next(r for r in rs if r['method']=='PPPM')
    ax=axs[1]
    for r,col in ((best,P.FAM_COL['uniform']),(old,P.FAM_COL['moderate'])):
        ax.bar(r['method'],r['median_matvec_s']*1e3,color=col)
    ax.set_ylabel('cached complete matvec (ms)');ax.set_title('(b) Common serial timing')
    ax=axs[2];x=np.arange(2);bw=.33
    ax.bar(x-bw/2,[best['median_config_setup_s']*1e3,old['median_config_setup_s']*1e3],bw,label='configuration')
    ax.bar(x+bw/2,[best['median_table_setup_s']*1e3,old['median_table_setup_s']*1e3],bw,label='parameter table')
    ax.set_xticks(x);ax.set_xticklabels(['B2','PPPM']);ax.set_ylabel('setup (ms)');ax.legend(fontsize=6)
    ax.set_title('(c) Separately measured setup')
    fig.tight_layout();save(fig,'fig4_optimization_cost')

def main():
    P.style();rows=X.read(X.RESULTS/'sweeps.csv')
    if rows:
        series=six(rows,'E_G')+six(rows,'E_GX')
        X.write(X.FIGURES/'plotted_series.csv',series);comparison(rows)
    optimization()

if __name__=='__main__':main()
