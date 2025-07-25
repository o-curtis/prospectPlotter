import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import glob
import corner
import prospect.io.read_results as reader 
from prospect.models.transforms import logsfr_ratios_to_sfrs
import matplotlib.gridspec as gridspec

plt.rcParams.update({'xtick.major.pad': '7.0'})
plt.rcParams.update({'xtick.major.size': '7.5'})
plt.rcParams.update({'xtick.major.width': '1.5'})
plt.rcParams.update({'xtick.minor.pad': '7.0'})
plt.rcParams.update({'xtick.minor.size': '3.5'})
plt.rcParams.update({'xtick.minor.width': '1.0'})
plt.rcParams.update({'ytick.major.pad': '7.0'})
plt.rcParams.update({'ytick.major.size': '7.5'})
plt.rcParams.update({'ytick.major.width': '1.5'})
plt.rcParams.update({'ytick.minor.pad': '7.0'})
plt.rcParams.update({'ytick.minor.size': '3.5'})
plt.rcParams.update({'ytick.minor.width': '1.0'})
plt.rcParams.update({'xtick.color': 'k'})
plt.rcParams.update({'ytick.color': 'k'})
plt.rcParams.update({'xtick.top': True})
#plt.rcParams.update({'font.size': 1})
plt.rcParams.update({'xtick.direction': 'in'})
plt.rcParams.update({'ytick.right': True})
plt.rcParams.update({'ytick.direction': 'in'})

class prospectorPlotter():
    def __init__(self, resFileName, tarName, numParams=19):
        
        self.numParams = numParams
        self.tarName = tarName
        
        # --- Load and build results, obs, model, and sps objects ---
        self.results, self.obs, self.model = reader.results_from(resFileName, dangerous=False)
        self.model = reader.get_model(self.results)
        self.sps   = reader.get_sps(self.results)
        
        # --- Get agebins ---
        self.agebins = self.results['model_params'][8]['init']
        
        # --- Get MAP Value (of the locations visited by the nested sampler) ---
        self.imax = np.argmax(self.results['lnprobability'])
        i, j = np.unravel_index(self.imax, self.results['lnprobability'].shape)
        self.theta_max = self.results['chain'][i, j, :].copy()
        
    def makeCorner(self, figName, datName, labels, trim=10000, thin=1, figsize=50, convertLogSFRToSFR=True, **extras):
        plt.clf()
        plt.close()
        sigma1 = 1. - np.exp(-(1./1.)**2/2.)
        sigma2 = 1. - np.exp(-(2./1.)**2/2.)
        sigma3 = 1. - np.exp(-(3./1.)**2/2.)
        
        chain = np.copy(self.results['chain'])
        if convertLogSFRToSFR:
            chain = self._convert_logsfr_ratios_to_sfrs()
        chain = chain[:, trim:chain.shape[1]:thin, :]
        flattened = chain.reshape(-1, chain.shape[2])
        #flattened = flattened[trim:flattened.shape[0]-trim:thin]
         
        fig = plt.figure(figsize=(figsize,figsize))
        print(flattened.shape)

        corner.corner(flattened, 
              labels=labels,
              fig=fig,
              levels=(sigma1, sigma2, sigma3),
              show_titles=True,
              fill_contours=True, 
              plot_density=True, 
              alpha=0.5,
              plot_datapoints=True,
              title_kwargs={'fontsize':18},
              label_kwargs={'fontsize':20})
              
        for ax in fig.get_axes():
            ax.tick_params(axis='both', labelsize=18)

        plt.savefig(figName, pad_inches=0.3)
        
        quantiles = []
        for k in range(flattened.shape[1]):
            quantiles.append(np.quantile(flattened[:,k], [0.16, 0.5, 0.84]))
        quantiles = np.array(quantiles)
        np.savetxt(datName, quantiles)
        
    def makeSFRt(self, figName, sfrtName, NBootstrap=10000, **extras):
        plt.clf()
        plt.close()
        logmass_MAP = self.theta_max[2]
        logsfr_ratios_MAP = self.theta_max[3:9]
        sfrs_MAP = logsfr_ratios_to_sfrs(logmass=logmass_MAP, 
                                         logsfr_ratios=logsfr_ratios_MAP,
                                         agebins=self.agebins)
        
        quantiles = self._get_confidence_intervals_on_sfh(N=NBootstrap)
        lower_quantiles = quantiles[0]
        sfh_median = quantiles[1]
        upper_quantiles = quantiles[2]
        
        agebins_plot = 10**self.agebins.flatten()
        print(agebins_plot)
        sfh_median_plot = [sfh_median[int(i/2)] for i in range(2*len(sfh_median))]
        lower_quantiles_plot = [lower_quantiles[int(i/2)] for i in range(2*len(lower_quantiles))]
        upper_quantiles_plot = [upper_quantiles[int(i/2)] for i in range(2*len(upper_quantiles))]
        
        fig, ax = plt.subplots(1,1, figsize=(16,8))
        ax.plot(agebins_plot, sfh_median_plot, linestyle='solid', color='black')
        ax.fill_between(agebins_plot, 
                        lower_quantiles_plot,
                        upper_quantiles_plot,
                        facecolor='tab:gray',
                        alpha=0.25)
        
        #prettify
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlabel(r"years", fontsize=12)
        ax.set_ylabel(r"SFR [$\rm M_\odot/yr$]", fontsize=12)
        ax.set_xlim(10**7, agebins_plot[-1])
        ax.set_ylim(7*10**-3, 1.5*10**2)
        ax.tick_params(which='both', labelsize=12)
#        ax.set_ylim(10**-2, 2*10**0)
        plt.savefig(figName)
        np.savetxt(sfrtName, quantiles)

    def makeSpectrum(self, figName, N=10000, **extras):
        plt.clf()
        plt.close()
        wspec = self.sps.wavelengths
        wphot = self.obs["phot_wave"]
        xmin, xmax = np.min(wphot)*0.8, np.max(wphot)/0.8
        ymin, ymax = self.obs["maggies"][self.obs['phot_mask']].min()*0.4, self.obs["maggies"][self.obs['phot_mask']].max()/0.2

        # --- Generate MAP Model ---
        prediction = self.model.mean_model(self.theta_max, obs=self.obs, sps=self.sps)
        pspec, pphot, pfrac = prediction
        
        errsModelPhot, errsModelSpec = self._get_confidence_interval_photometry_and_spectra(N=N)

        fig = plt.figure(figsize=(18,10))
        gs = gridspec.GridSpec(3, 1)
        ax_main = plt.subplot(gs[0:2,0])
        ax_resid= plt.subplot(gs[2,0], sharex=ax_main)
        fig.subplots_adjust(wspace=0, hspace=0)
        gs.update(hspace=0)
        ax_main.get_xaxis().set_visible(False)
        ax_main.set_yscale('log')
        ax_main.set_xscale('log')

        ax_main.fill_between(wspec, 
                           errsModelSpec[0], 
                           errsModelSpec[2], 
                           color='tab:gray', 
                           alpha=0.7,
                           label='Model Spectrum')

        ax_main.errorbar(wphot[self.obs['phot_mask']], pphot[self.obs['phot_mask']], label='Model photometry',
                       marker='s', markersize=10, alpha=0.8, ls='', lw=3,
                       markerfacecolor='none', markeredgecolor='slateblue',
                       markeredgewidth=3)

        ax_main.errorbar(wphot[self.obs['phot_mask']], self.obs['maggies'][self.obs['phot_mask']], 
        yerr=self.obs['maggies_unc'][self.obs['phot_mask']],
                       label='Observed photometry',
                       marker='o', markersize=10, alpha=0.8, ls='', lw=3,
                       ecolor='tomato', markerfacecolor='none', markeredgecolor='tomato',
                       markeredgewidth=3)
             
        print(errsModelPhot.shape)
             
        O, E, dO, dE = pphot[self.obs['phot_mask']], self.obs['maggies'][self.obs['phot_mask']], \
                        errsModelPhot[1][self.obs['phot_mask']]-errsModelPhot[0][self.obs['phot_mask']], \
                        self.obs['maggies_unc'][self.obs['phot_mask']]
        chiSqPhot = (pphot[self.obs['phot_mask']] - self.obs['maggies'][self.obs['phot_mask']])**2 \
                    / self.obs['maggies'][self.obs['phot_mask']]
        errChiSqPhot = np.sqrt( (((O-E)*2/E)*dO)**2 + (((-1*(2*(O-E)*E)-(O-E)**2)/E**2)*dE)**2 )
        ax_resid.errorbar(wphot[self.obs['phot_mask']], chiSqPhot, errChiSqPhot, marker='s', markersize=10, alpha=0.8, ls='', lw=3,\
                       mfc='black', capsize=3, markeredgecolor='black', ecolor='black', markeredgewidth=5)

    # Prettify
        ax_resid.set_xlabel(r'Wavelength [$\rm \AA$]', fontsize=14)

        ax_main.set_ylabel('Flux Density [maggies]', fontsize=14)
        ax_resid.set_ylabel(r"$\rm \chi^2$", fontsize=14)
        ax_main.set_xlim([xmin, 5*10**5])
        ax_main.set_ylim([5*10**-9, 5*10**-3])
        #ax_main.set_ylim(10**-12, 10**-6)
        #ax_main.set_ylim(ymin, ymax)
        ax_main.legend(loc='best', fontsize='x-small')
        
        ax_main.tick_params(which='both', labelsize=14)
        ax_resid.tick_params(which='both', labelsize=14)
        ax_resid.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax_resid.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        
        plt.ticklabel_format(style='plain', axis='y')
        plt.tight_layout()
        plt.savefig(figName)
        
    def makeTrace(self, figName, **extras):
        #make trace plot
        plt.clf()
        plt.close()
        tracefig = reader.traceplot(self.results, figsize=(20,10))
        chosen = np.random.choice(self.results['run_params']["nwalkers"], size=10, replace=False)
        plt.savefig(figName)
       
    def _convert_logsfr_ratios_to_sfrs(self, **extras):
        
        chain = np.copy(self.results['chain'])
        logsfr_ratios = chain[:,:,3:9] #6 ratios, 7 total sfrs
        logmass = chain[:,:,2]
        sfrs = np.zeros_like(chain[:,:,3:10]) #grab one more than there are ratios
        
        for walkerix in range(logsfr_ratios.shape[0]):
            for iterationix in range(logsfr_ratios.shape[1]):
                thismass = logmass[walkerix, iterationix]
                these_logsfr_ratios = logsfr_ratios[walkerix, iterationix, :]
                these_sfrs = logsfr_ratios_to_sfrs(logmass=thismass, logsfr_ratios=these_logsfr_ratios, agebins=self.agebins)
                sfrs[walkerix, iterationix, :] = these_sfrs
        chain[:,:,3:9] = sfrs[:, :, :-1]
        chain_expanded = np.insert(chain, 9, sfrs[:,:,-1], axis=2)
        print(chain_expanded[0,0,:])
        print(chain_expanded.shape)
        
        return chain_expanded
        
    def _get_confidence_interval_photometry_and_spectra(self, **extras):
        randint = np.random.randint
        nwalkers, niter = self.results['run_params']['nwalkers'], self.results['run_params']['niter']
        tmpsetPhot=[]
        tmpsetSpec=[]
        for _ in range(extras['N']):
            theta = self.results['chain'][randint(nwalkers), randint(niter)]
            prediction = self.model.mean_model(theta, obs=self.obs, sps=self.sps)
            pspec, pphot, pfrac = prediction
            tmpsetPhot.append(pphot)
            tmpsetSpec.append(pspec)
        tmpsetPhot = np.array(tmpsetPhot)
        tmpsetSpec = np.array(tmpsetSpec)
        errsPhot = np.quantile(tmpsetPhot, [0.16, 0.5, 0.84], axis=0)
        errsSpec = np.quantile(tmpsetSpec, [0.16, 0.5, 0.84], axis=0)
        return errsPhot, errsSpec


    def _get_confidence_intervals_on_sfh(self, **extras):
        randint = np.random.randint
        nwalkers, niter = self.results['run_params']['nwalkers'], self.results['run_params']['niter']
        tmpset=[]
        for _ in range(extras['N']):
            theta = self.results['chain'][randint(nwalkers), randint(niter)]
            logsfr_ratios = theta[3:9]
            logmass = theta[2]
            sfrs = logsfr_ratios_to_sfrs(logmass=logmass, logsfr_ratios=logsfr_ratios, agebins=self.agebins)
            tmpset.append(sfrs)
        tmpset = np.array(tmpset)
        errs = np.quantile(tmpset, [0.16, 0.5, 0.84], axis=0)
        return errs


if __name__ == '__main__':
    labels= ['zsol', 'dust2', 'logmass', 'sfr1', 'sfr2', 'sfr3', 'sfr4', 'sfr5', 'sfr6', 'sfr7', 'eUmin', 'eqpah', 'egamma', 'dustIndex', 'dustRatio', 'fagn', 'agnTau', 'DSalpha', 'T_BB']
    
    import glob
    tar = 
    base = "/storage/group/jtw13/default/oCurtisWorkDir/ghat/prospectTest/"
    BASE_DIR = base+str(tar)+"/"
    RESULTS_FILE = glob.glob(BASE_DIR+"*.h5")[0]
    
    thisPlotter = prospectorPlotter(RESULTS_FILE, tar)
    thisPlotter.makeCorner(figName=BASE_DIR+"corner.pdf",trim=0, datName=BASE_DIR+"quantiles.dat", labels=labels)
    thisPlotter.makeSpectrum(figName=BASE_DIR+"spectrum.pdf", N=1)
    thisPlotter.makeTrace(figName=BASE_DIR+"trace.pdf")
    thisPlotter.makeSFRt(figName=BASE_DIR+"SFRt.pdf", sfrtName=BASE_DIR+"SFRt.dat")
    thisPlotter = None
    print("Done with ", TAR_NAME)
