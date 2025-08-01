import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import glob
import corner
import prospect.io.read_results as reader 
from prospect.models.transforms import logsfr_ratios_to_sfrs
import matplotlib.gridspec as gridspec
from prospectPlotter.helpers import maggies_to_lsun_per_hz, wav_to_nu

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

class prospectPlotter():
    def __init__(self, resFileName, numParams=19):
        
        self.numParams = numParams
        
        # --- Load and build results, obs, model, and sps objects ---
        self.results, self.obs, self.model = reader.results_from(resFileName, dangerous=False)
        self.model = reader.get_model(self.results)
        self.sps   = reader.get_sps(self.results)
        
        # --- Grab the redshift ---
        self.zred = self.results['model_params'][0]['init']
        
        # --- Get agebins ---
        self.agebins = self.results['model_params'][8]['init']
        
        # --- Get MAP Value (of the locations visited by the nested sampler) ---
        self.imax = np.argmax(self.results['lnprobability'])
        i, j = np.unravel_index(self.imax, self.results['lnprobability'].shape)
        self.theta_max = self.results['chain'][i, j, :].copy()
        
        # --- Generate MAP Model ---
        prediction = self.model.mean_model(self.theta_max, obs=self.obs, sps=self.sps)
        self.pspec, self.pphot, self.pfrac = prediction
        self.pphot = self.pphot[self.obs['phot_mask']]
        
        # --- Get wavelengths ---
        self.wspec = self.sps.wavelengths
        self.wphot = self.obs["phot_wave"][self.obs['phot_mask']]
        
        # --- Get observed photometry ---
        self.ophot = self.obs["maggies"][self.obs['phot_mask']]
        self.ophoterr = self.obs['maggies_unc'][self.obs['phot_mask']]
        
    def makeCorner(self, figName, datName, labels=None, trim=10000, thin=1, figsize=50, convertLogSFRToSFR=False, **extras):
        # clean the figure
        plt.clf()
        plt.close()

        if labels==None:
                labels = self.results["theta_labels"]
        
        # sigma levels for contours
        sigma1 = 1. - np.exp(-(1./1.)**2/2.)
        sigma2 = 1. - np.exp(-(2./1.)**2/2.)
        sigma3 = 1. - np.exp(-(3./1.)**2/2.)
        
      	# get the chain
      	# convert logsfr_ratios to sfrs
      	# trim, thin, and flatten the chain
        chain = np.copy(self.results['chain'])
        if convertLogSFRToSFR:
            chain = self._convert_logsfr_ratios_to_sfrs()
        chain = chain[:, trim:chain.shape[1]:thin, :]
        flattened = chain.reshape(-1, chain.shape[2])
         
        # create the figure 
        fig = plt.figure(figsize=(figsize,figsize))

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
            
        # --- Prettify ---      
        for ax in fig.get_axes():
            ax.tick_params(axis='both', labelsize=18)
        plt.savefig(figName, pad_inches=0.3)
        
        # Saves the [0.16, 0.5, 0.84] quantiles for separate analysis
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

    def makeSpectrum(self, figName, N=10000, style=0, **extras):
        # clean up
        plt.clf()
        plt.close()

        # get confidence intervals on photometry and spectra
        errsModelPhot, errsModelSpec = self._get_confidence_interval_photometry_and_spectra(N=N)
        errsModelPhot = np.abs(errsModelPhot[[0,2], :] - errsModelPhot[1])

        # create figure
        # ax_main shows spectra/photom
        # ax_resid shows chi-sq values
        fig = plt.figure(figsize=(18,10))
        gs = gridspec.GridSpec(3, 1)
        ax_main = plt.subplot(gs[0:2,0])
        ax_resid= plt.subplot(gs[2,0], sharex=ax_main)
        fig.subplots_adjust(wspace=0, hspace=0)
        gs.update(hspace=0)
             
        # getting reduced chi-sq between model and observed values
        O, E, dO, dE = self.pphot, self.ophot, \
                       errsModelPhot[1][self.obs['phot_mask']]-errsModelPhot[0][self.obs['phot_mask']], \
                       self.obs['maggies_unc'][self.obs['phot_mask']]
        chiSqPhot = (O - E)**2 / O
        errChiSqPhot = np.sqrt( (((O-E)*2/E)*dO)**2 + (((-1*(2*(O-E)*E)-(O-E)**2)/E**2)*dE)**2 )
        
        # --- Setting style options for plotting ---
        if style==0: #i.e., logmaggies Vs. loglambda
            wspec = self.wspec
            wphot = self.wphot
            pphot = self.pphot
            ophot = self.ophot
            ophoterr = self.ophoterr
            errsModelSpec[0] = errsModelSpec[0]
            errsModelSpec[1] = errsModelSpec[1]
            errsModelPhot = errsModelPhot
            ax_main.set_yscale('log')
            ax_main.set_ylabel('Flux Density [maggies]', fontsize=14)
            ax_resid.set_xlabel(r'Wavelength [$\rm \AA$]', fontsize=14)
            ax_main.set_ylim([5*10**-9, 5*10**-3])

        elif style==1: #i.e., nuL_nu Vs. lognu
            wspec = wav_to_nu(self.wspec)
            wphot = wav_to_nu(self.wphot)
            zred = np.array([self.zred])
            pphot = maggies_to_lsun_per_hz(self.pphot, zred=zred)*wphot
            ophot = maggies_to_lsun_per_hz(self.ophot, zred=zred)*wphot
            pphot=pphot[0]
            ophot=ophot[0]
            ophoterr = maggies_to_lsun_per_hz(self.ophoterr, zred=zred)
            errsModelSpec[0] = maggies_to_lsun_per_hz(errsModelSpec[0], zred=zred)*wspec
            errsModelSpec[1] = maggies_to_lsun_per_hz(errsModelSpec[1], zred=zred)*wspec
            errsModelPhot = maggies_to_lsun_per_hz(errsModelPhot, zred=zred)*wphot
            print(wphot)
            print(ophot)
            ax_main.set_ylim(0, 0.001)
            ax_main.set_xlim(1.5*10**10, 6*10**16)
            ax_main.set_ylabel(r'$\rm \nu L_{\nu}$ [$\rm L_{\odot}$]', fontsize=14)
            ax_resid.set_xlabel(r'$\rm \nu$ [Hz]', fontsize=14)
        
        # --- Plotting ---
        ax_main.fill_between(wspec, 
                             errsModelSpec[0], 
                             errsModelSpec[2], 
                             color='tab:gray', 
                             alpha=0.7,
                             label='Model Spectrum')

        print(np.shape(wphot), np.shape(pphot))
        ax_main.errorbar(wphot, pphot, yerr=errsModelPhot, label='Model photometry',
                         marker='s', markersize=10, alpha=0.8, ls='', lw=3,
                         markerfacecolor='none', markeredgecolor='slateblue',
                         markeredgewidth=3)

        ax_main.errorbar(wphot, ophot, 
                         yerr=ophoterr,
                         label='Observed photometry',
                         marker='o', markersize=10, alpha=0.8, ls='', lw=3,
                         ecolor='tomato', markerfacecolor='none', markeredgecolor='tomato',
                         markeredgewidth=3)
        
        ax_resid.errorbar(wphot, chiSqPhot, 
                          errChiSqPhot, marker='s', markersize=10, 
                          alpha=0.8, ls='', lw=3,
                          mfc='black', capsize=3, markeredgecolor='black', 
                          ecolor='black', markeredgewidth=5)

        # --- Prettify ---
        #get bounds
        xmin, xmax = np.min(wphot)*0.8, np.max(wphot)/0.8
        ymin, ymax = ophot*0.4, ophot.max()/0.2
        
        ax_main.get_xaxis().set_visible(False)
        ax_main.set_xscale('log')
        #ax_main.set_xlim([xmin, 5*10**5])
        

        ax_resid.set_ylabel(r"$\rm \chi^2$", fontsize=14)
        ax_main.legend(loc='best', fontsize='x-small')
        
        ax_main.tick_params(which='both', labelsize=14)
        ax_resid.tick_params(which='both', labelsize=14)
        ax_resid.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax_resid.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
        
        plt.ticklabel_format(style='plain', axis='y')
        plt.tight_layout()
        plt.savefig(figName)
        
    def makeTrace(self, figName, **extras):
        # make trace plot
        # this is just using prospectors old traceplot function
        # Will update this eventually to look nicer
        plt.clf()
        plt.close()
        tracefig = reader.traceplot(self.results, figsize=(20,10))
        chosen = np.random.choice(self.results['run_params']["nwalkers"], size=10, replace=False)
        plt.savefig(figName)
       
    def _convert_logsfr_ratios_to_sfrs(self, **extras):
        '''
        Pulls the chain from the results file.
        Converts the logsfr entries to straight
        star formation rates for ease of analysis.
        Note this extends the chain by one dimension 
        along the 3rd axis as we expand all ratios
        into SRRs.
    
        '''
        default_labels = self.results['theta_labels']
        logsfr_indices = [i for i, s in enumerate(default_labels) if 'logsfr' in s]
        min_ix, max_ix = logsfr_indices[0], logsfr_indices[-1]
        logmass_ix = [i for i, s in enumerate(default_labels) if "logmass" in s][0]
        
        chain = np.copy(self.results['chain'])
        logsfr_ratios = chain[:,:,min_ix:max_ix+1] #6 ratios, 7 total sfrs
        logmass = chain[:,:,logmass_ix]
        sfrs = np.zeros_like(chain[:,:,min_ix:max_ix+2]) #grab one more than there are ratios
        
        for walkerix in range(logsfr_ratios.shape[0]):
            for iterationix in range(logsfr_ratios.shape[1]):
                thismass = logmass[walkerix, iterationix]
                these_logsfr_ratios = logsfr_ratios[walkerix, iterationix, :]
                these_sfrs = logsfr_ratios_to_sfrs(logmass=thismass, logsfr_ratios=these_logsfr_ratios, agebins=self.agebins)
                sfrs[walkerix, iterationix, :] = these_sfrs
        chain[:,:,min_ix:max_ix+1] = sfrs[:, :, :-1]
        chain_expanded = np.insert(chain, max_ix+1, sfrs[:,:,-1], axis=2)
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
 
    import os   
    RESULTS_FILE = "/storage/group/jtw13/default/oCurtisWorkDir/ghat/prospectTest/NGC_3265/_25Jul22-16.09_result.h5"
    #RESULTS_FILE = "/storage/group/jtw13/default/oCurtisWorkDir/ghat/prospectTest/NGC_3265_50/_25Jul22-16.09_result.h5"
    BASE_DIR = os.path.dirname(RESULTS_FILE) + os.sep
    
    thisPlotter = prospectorPlotter(RESULTS_FILE)
    #thisPlotter.makeCorner(figName=BASE_DIR+"corner.pdf",trim=0, datName=BASE_DIR+"quantiles.dat", labels=labels)
    thisPlotter.makeSpectrum(figName=BASE_DIR+"spectrum.pdf", N=1000)
    #thisPlotter.makeTrace(figName=BASE_DIR+"trace.pdf")
    #thisPlotter.makeSFRt(figName=BASE_DIR+"SFRt.pdf", sfrtName=BASE_DIR+"SFRt.dat")
    thisPlotter = None
