#!/usr/bin/env python
import os
os.environ["OMP_NUM_THREADS"] = "8"
import  sys
import argparse
import glob
import time
import yaml
import numpy as np

# Create a parser for first handling the config file
cfg_parser = argparse.ArgumentParser(
    description='Generate sky simulations', add_help=False
)
cfg_parser.add_argument(
    '-c',
    '--config-file',
    action='store',
    help='YAML file containing configuration parameters for sims. '
    'Any of the arguments below can be included in the file, and '
    'will be overridden if supplied at the command line.',
)
args, _ = cfg_parser.parse_known_args()

# https://stackoverflow.com/a/20422915
class ActionNoYes(argparse.Action):
    def __init__(self, option_strings, dest, default=None, required=False, help=None):

        if default is None:
            raise ValueError('You must provide a default with Yes/No action')
        if len(option_strings) != 1:
            raise ValueError('Only single argument is allowed with YesNo action')
        opt = option_strings[0]
        if not opt.startswith('--'):
            raise ValueError('Yes/No arguments must be prefixed with --')

        opt = opt[2:]
        opts = ['--' + opt, '--no-' + opt]
        super(ActionNoYes, self).__init__(
            opts,
            dest,
            nargs=0,
            const=None,
            default=default,
            required=required,
            help=help,
        )

    def __call__(self, parser, namespace, values, option_strings=None):
        if option_strings.startswith('--no-'):
            setattr(namespace, self.dest, False)
        else:
            setattr(namespace, self.dest, True)


parser = argparse.ArgumentParser(
    parents=[cfg_parser], formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
group = parser.add_argument_group('Configuration parameters')
group.add_argument(
    '--output-root',
    '-o',
    action='store',
    required=True,
    help='Output directory for storing sims data files',
)
group.add_argument(
    '--num-threads', action='store', type=int, default=1, help='number of threads'
)
group.add_argument(
    '--sim-index',
    action='store',
    type=int,
    default=0,
    help='Index of first sim to generate. The random seed is uniquely '
    'determined for each foreground component from the sim number, so '
    'use this parameter to add more sims to an existing database.',
)
group.add_argument(
    '--num-sims',
    action='store',
    type=int,
    default=1,
    help='Number of simulations.  Sims will be numbered from '
    'sim_index to sim_index + num_sims',
)
group.add_argument(
    '--freqs',
    action='store',
    type=int,
    nargs='+',
    choices=[90, 150, 220],
    default=[150],
    help='Frequency channel(s) to include. Important for foregrounds '
    'and instrumental beam/noise.',
)
group.add_argument(
    '--nside',
    action='store',
    type=int,
    default=512,
    help='Healpix map resolution parameter',
)
group.add_argument(
    '--lmax',
    action='store',
    type=int,
    default=1000,
    help='Maximum multipole (ell) to simulate',
)
group.add_argument(
    '--pol', action=ActionNoYes, default=True, help='Include polarization (Q + U)'
)
group.add_argument('--cmb', action=ActionNoYes, default=False, help='Make cmb skies')
group.add_argument(
    '--foregrounds', action=ActionNoYes, default=False, help='Make foreground skies'
)
group.add_argument('--noise', action=ActionNoYes, default=False, help='Make noise sims')
group.add_argument(
    '--combine-gaussian-fg',
    action=ActionNoYes,
    default=False,
    help='Combine gaussian foregrounds into one file per frequency per sim.',
)
group.add_argument(
    '--beam',
    action=ActionNoYes,
    default=False,
    help='Smooth the final combined map with the beam',
)
group.add_argument(
    '--mask-file', action='store', help='Path to sky mask to apply to the final map(s)'
)
group.add_argument(
    '--debug',
    default=False,
    action=ActionNoYes,
    help='Show plots of the sim CMB and foregrounds.',
)
group.add_argument(
    '--verbose',
    default=False,
    action=ActionNoYes,
    help='Print healpy outputs, reduce log level',
)

group = parser.add_argument_group('CMB parameters')
group.add_argument(
    '--camb-file',
    default=os.path.join(
        'planck18_TTEEEE_lowl_lowE_lensing',
        'base_plikHM_TTTEEE_lowl_lowE_lensing_lensedCls.dat',
    ),
    help='CAMB power spectra to use for CMB simulations',
)

group = parser.add_argument_group('Beam parameters')
group.add_argument(
    '--beam-file',
    action='store',
    help='Path to file containing B_ell for each frequency',
)
group.add_argument(
    '--fwhm-90',
    action='store',
    type=float,
    default=1.7,
    help='Default FWHM in arcmin for 90 GHz band, '
    'used for a gaussian beam if beam file is not supplied',
)
group.add_argument(
    '--fwhm-150',
    action='store',
    type=float,
    default=1.4,
    help='Default FWHM in arcmin for 150 GHz band, '
    'used for a gaussian beam if beam file is not supplied',
)
group.add_argument(
    '--fwhm-220',
    action='store',
    type=float,
    default=1.2,
    help='Default FWHM in arcmin for 220 GHz band, '
    'used for a gaussian beam if beam file is not supplied',
)

group = parser.add_argument_group('Foreground parameters')
group.add_argument(
    '--gaussian-fg-model',
    action='store',
    default='reichardt',
    choices=['george', 'reichardt'],
    help='Gaussian foreground model to use',
)
group.add_argument(
    '--rescale',
    action='store',
    default='False',
    help='rescale the fg power spectrum to 3G effective frequencies',
)
group.add_argument(
    '--gaussian-thermal-sz',
    action=ActionNoYes,
    default=True,
    help='Include a gaussian thermal SZ foreground component, '
    'using values from SPT-SZ (arXiv: 1408.3161)',
)
group.add_argument(
    '--gaussian-kinetic-sz',
    action=ActionNoYes,
    default=True,
    help='Include a gaussian kinetic SZ foreground component, '
    'using values from SPT-SZ (arXiv: 1408.3161)',
)
group.add_argument(
    '--gaussian-radio-galaxies',
    action=ActionNoYes,
    default=True,
    help='Include a gaussian radio galaxy foreground component, '
    'using values from SPT-SZ (arXiv: 1408.3161)',
)
group.add_argument(
    '--gaussian-dusty-galaxies',
    action=ActionNoYes,
    default=True,
    help='Include a gaussian dusty galaxy foreground component, '
    'using values from SPT-SZ (arXiv: 1408.3161)',
)
group.add_argument(
    '--gaussian-dg-clustering',
    action=ActionNoYes,
    default=True,
    help='Include the clustering term in the gaussian dusty galaxy '
    'foreground component, using values from SPT-SZ (arXiv: 1408.3161). '
    'Only used if --gaussian-dusty-galaxies is supplied.',
)
group.add_argument(
    '--poisson-radio-galaxies',
    action=ActionNoYes,
    default=True,
    help='Include poisson radio galaxy foreground component.',
)
group.add_argument(
    '--galactic-dust',
    action=ActionNoYes,
    default=False,
    help='Include galactic dust using BK15 numbers.',
)
group.add_argument(
    '--poisson-rg-model',
    action='store',
    default='dezotti',
    choices=['dezotti'],
    help='Model giving source counts and fluxes for radio galaxies.',
)
group.add_argument(
    '--poisson-dusty-galaxies',
    action=ActionNoYes,
    default=True,
    help='Include poisson dusty galaxy foreground component.',
)
group.add_argument(
    '--poisson-dg-model',
    action='store',
    default='bethermin',
    choices=['bethermin'],
    help='Model giving source counts and fluxes for dusty galaxies.',
)
group.add_argument(
    '--detected-point-sources',
    action=ActionNoYes,
    default=False,
    help='Include detected point sources instead of the poisson ones.',
)
group.add_argument(
    '--detected-spt-clusters',
    action=ActionNoYes,
    default=False,
    help='Include detected point sources instead of the poisson ones.',
)
group.add_argument(
    '--ptsrc-file',
    action='store',
    default="spt3g_1500d_source_list_zp_Feb24_2020.txt",
    help='Path to the ptsrc list with locations and fluxes',
)
group.add_argument(
    '--cluster-file',
    action='store',
    default="spt3g_1500d_cluster_list_lb_Feb17_2020.txt",
    help='Path to the cluster list with location and sn',
)
group.add_argument(
    '--correct-spectra',
    action=ActionNoYes,
    default=False,
    help='Correct the foreground spectra if fainter sources are populated',
)
group.add_argument(
    '--correction-file',
    action='store',
    default="reichardt_fg_spectra_correction.pkl",
    help='Path to file with foreground spectra corrections',
)
group.add_argument(
    '--detected-min-flux',
    action='store',
    type=float,
    default=0.0,
    help='Lowest flux for the detected sources at 150',
)
group.add_argument(
    '--detected-max-flux',
    action='store',
    type=float,
    default=1e10,
    help='Max flux for the detected sources at 150',
)
group.add_argument(
    '--detected-cluster-min-sn',
    action='store',
    type=float,
    default=5.0,
    help='Lowest sn for the detected clusters',
)
group.add_argument(
    '--detected-cluster-max-sn',
    action='store',
    type=float,
    default=1e10,
    help='Max sn for the detected clusters',
)
group.add_argument(
    '--sz-pol-fraction',
    action='store',
    type=float,
    default=0.0,
    help='Polarization fraction of SZ components.',
)
group.add_argument(
    '--dg-pol-fraction',
    action='store',
    type=float,
    default=0.02,
    help='Polarization fraction of dusty galaxy component, '
    'Default from arXiv: astro-ph/0610485',
)
group.add_argument(
    '--rg-pol-fraction',
    action='store',
    type=float,
    default=0.03,
    help='Polarization fraction of radio galaxy component, '
    'Default from ACTPol: arXiv: 1811.01854; '
    'SPTpol: Gupta et al. in prep., etc.',
)
group.add_argument(
    '--min-flux-limit',
    action='store',
    type=float,
    default=6.4e-3,
    help='Poisson minimum source flux in Janskys.  Must agree with '
    'value used to calculate foreground power spectra.',
)
group.add_argument(
    '--max-flux-limit',
    action='store',
    type=float,
    default=5.0e-2,
    help='Poission maximum source flux in Janskys.',
)
group.add_argument(
    '--spec-index-radio-90-150',
    action='store',
    type=float,
    default=-0.7,
    help='Power-law index for scaling radio source flux at 150 GHz'
    'to 90 GHz, default from SPT-SZ W. Everett paper Fig. 3',
)
group.add_argument(
    '--spec-index-radio-220-150',
    action='store',
    type=float,
    default=-0.6,
    help='Power-law index for scaling radio source flux at 150 GHz '
    'to 220 GHz, default from SPT-SZ W. Everett paper Fig. 3',
)
group.add_argument(
    '--spec-index-dust-90-150',
    action='store',
    type=float,
    default=3.4,
    help='Power-law index for scaling dusty source flux at 150 GHz '
    'to 90 GHz, default from SPT-SZ W. Everett paper Fig. 3',
)
group.add_argument(
    '--spec-index-dust-220-150',
    action='store',
    type=float,
    default=3.4,
    help='Power-law index for scaling dusty source flux at 150 GHz '
    'to 220 GHz, default from SPT-SZ W. Everett paper Fig. 3',
)

group = parser.add_argument_group('Noise parameters')
group.add_argument(
    '--delta-t-90',
    action='store',
    type=float,
    default=3.0,
    help='Map depth at 90 GHz in uK-arcmin, default is ' 'SPT-3G 5-year forecast',
)
group.add_argument(
    '--delta-t-150',
    action='store',
    type=float,
    default=2.2,
    help='Map depth at 150 GHz in uK-arcmin, default is ' 'SPT-3G 5-year forecast',
)
group.add_argument(
    '--delta-t-220',
    action='store',
    type=float,
    default=8.8,
    help='Map depth at 220 GHz in uK-arcmin, default is ' 'SPT-3G 5-year forecast',
)
group.add_argument(
    '--weight-map-90', action='store', help='Path to 90 GHz weight map file',
)
group.add_argument(
    '--weight-map-150', action='store', help='Path to 150 GHz weight map file',
)
group.add_argument(
    '--weight-map-220', action='store', help='Path to 220 GHz weight map file',
)
group.add_argument(
    '--rho-oneoverf-noise',
    action='store',
    type=float,
    default=1.0,
    help='Correlation coefficient for 1/f noise between different bands',
)
group.add_argument(
    '--lknee-t-90',
    action='store',
    type=int,
    default=1200,
    help='Multipole where 1/f noise flattens out in '
    'temperature at 90 GHz, default based on SPT-SZ',
)
group.add_argument(
    '--lknee-t-150',
    action='store',
    type=int,
    default=2200,
    help='Multipole where 1/f noise flattens out in '
    'temperature at 150 GHz, default based on SPT-SZ',
)
group.add_argument(
    '--lknee-t-220',
    action='store',
    type=int,
    default=2300,
    help='Multipole where 1/f noise flattens out in '
    'temperature at 220 GHz, default based on SPT-SZ',
)
group.add_argument(
    '--lknee-p-90',
    action='store',
    type=int,
    default=300,
    help='Multipole where 1/f noise flattens out in '
    'polarization at 90 GHz, default is a conservative guess',
)
group.add_argument(
    '--lknee-p-150',
    action='store',
    type=int,
    default=300,
    help='Multipole where 1/f noise flattens out in '
    'polarization at 150 GHz, default is a conservative guess',
)
group.add_argument(
    '--lknee-p-220',
    action='store',
    type=int,
    default=300,
    help='Multipole where 1/f noise flattens out in '
    'polarization at 220 GHz, default is a conservative guess',
)
group.add_argument(
    '--alphaknee-t-90',
    action='store',
    type=float,
    default=3.0,
    help='slope for 1/f noise temperature at 90 GHz, default comes from SPT-SZ',
)
group.add_argument(
    '--alphaknee-t-150',
    action='store',
    type=float,
    default=4.0,
    help='slope for 1/f noise temperature at 150 GHz, default comes from SPT-SZ',
)
group.add_argument(
    '--alphaknee-t-220',
    action='store',
    type=float,
    default=4.0,
    help='slope for 1/f noise temperature at 220 GHz, default comes from SPT-SZ',
)
group.add_argument(
    '--alphaknee-p-90',
    action='store',
    type=float,
    default=1.0,
    help='slope for 1/f noise'
    'polarization at 90 GHz, default is a conservative guess',
)
group.add_argument(
    '--alphaknee-p-150',
    action='store',
    type=float,
    default=1.0,
    help='slope for 1/f noise'
    'polarization at 150 GHz, default is a conservative guess',
)
group.add_argument(
    '--alphaknee-p-220',
    action='store',
    type=float,
    default=1.0,
    help='slope for 1/f noise'
    'polarization at 220 GHz, default is a conservative guess',
)
group.add_argument(
    '--scaled-pspectra',
    action=ActionNoYes,
    default=True,
    help='If True, obtain the foreground spectra for the'
    'specified band by scaling the SPT-SZ reference band (band0) spectra',
)
group.add_argument(
    '--band0',
    action='store',
    default='150GHz',
    type=str,
    choices=['90GHz', '150GHz', '220GHz'],
    help='reference frequency band at which the '
    'SPT foreground spectra must be queried for foreground scalings.',
)
group.add_argument(
    '--use-spt-datapoints',
    action=ActionNoYes,
    default=True,
    help='True: Use SPT-SZ/SPTpol G15/R20 median data points for foreground amplitudes.'
    'False: Use SPT-SZ/SPTpol G15/R20 best-fit values for foreground amplitudes.',
)
group.add_argument(
    '--dg-clus-template-id',
    action='store',
    type=int,
    choices=[0, 1],
    default=0,
    help='template for dg-cl'
    '0: contribution is split into 1- and 2-halo terms.'
    '1: D_{\ell} \propto \ell^0.8.',
)
group.add_argument(
    '--spec-index-rg',
    action='store',
    type=float,
    default=-0.76,
    help='Spectral index for dusty galaxies (DG) Poisson component'
    '(c.f. Eq. 13 of G15 https://arxiv.org/pdf/1408.3161.pdf)'
    'Default value is 1.48 from R20 (sec 6.1.1 https://arxiv.org/pdf/2002.06197.pdf).'
    'For G15 value check Sec 6.1.1 of G15 https://arxiv.org/pdf/1408.3161.pdf.',
)
group.add_argument(
    '--spec-index-dg-po',
    action='store',
    type=float,
    default=1.48,
    help='Spectral index for dusty galaxies (DG) Poisson component'
    '(c.f. Eq. 13 of G15 https://arxiv.org/pdf/1408.3161.pdf)'
    'Default value is 1.48 from R20 (sec 6.1.1 https://arxiv.org/pdf/2002.06197.pdf).'
    'For G15 value check Sec 6.1.1 of G15 https://arxiv.org/pdf/1408.3161.pdf.',
)
group.add_argument(
    '--spec-index-dg-clus',
    action='store',
    type=float,
    default=2.23,
    help='Spectral index for dusty galaxies (DG) clustering component '
    '(c.f. Eq. 13 of G15 https://arxiv.org/pdf/1408.3161.pdf).'
    'Default value is 2.23 from R20 (sec 6.1.1 https://arxiv.org/pdf/2002.06197.pdf).'
    'For G15 value check Sec 6.1.1 of G15 https://arxiv.org/pdf/1408.3161.pdf.',
)
group.add_argument(
    '--cib-temp',
    action='store',
    type=float,
    default=25.0,
    help='Dust Blackbody temperature in Kelvin. Will be converted into G3units in the code.'
    'Default value is R20 value cib_temp = 25 kelvin.'
    'Cannot find the reference for this in https://arxiv.org/pdf/2002.06197.pdf.'
    'For G15 it is 20 kelvin'
    '(Sec 5.4.3 of G15 https://arxiv.org/pdf/1408.3161.pdf).',
)


# override default values from the config file
if args.config_file:
    cfg = yaml.safe_load(open(args.config_file, 'r'))
    parser.set_defaults(**cfg)

# parse all the arguments now
args = parser.parse_args()

# set number of threads before importing anything else
os.putenv('OMP_NUM_THREADS', str(args.num_threads))

# import all the big packages
import numpy as np
import healpy as hp
from spt3g import core
from spt3g.simulations import sim_tools, cmb, instrument, foregrounds

# for basic debugging purposes we will make simulations of 90 and 150.
if args.debug:
    args.freqs = [90, 150]

# print timestamps in log messages
core.G3Logger.global_logger.timestamps = True
if args.verbose:
    core.set_log_level(core.G3LogLevel.LOG_INFO)

# standardize file structure relative to output_root
def get_filename(subdir, name, sim_num, freq=None):
    if freq is not None:
        # ensure a unique filename
        name = name.replace('_alms', '_%dghz_alms' % freq)
        name = name.replace('_map', '_%dghz_map' % freq)
        if 'ghz' not in name:
            name = '%s_%dghz' % (name, freq)
    # ordered alphabetically by sim number using zero padding
    filename = os.path.join(args.output_root, subdir, '%s_%04d.fits' % (name, sim_num))
    # ensure output directory exists
    save_dir = os.path.dirname(filename)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return filename


# ===========================================================
# Set up simulation parameters
# ===========================================================
# list of sim numbers to generate
sim_range = list(range(args.sim_index, args.sim_index + args.num_sims))

# Set up dictionary to keep track of alms made for each sim.
# These will be added into a combined map if requested.
coadd_alms = {}
for freq in args.freqs:
    coadd_alms[freq] = {}
    for sim_num in sim_range:
        coadd_alms[freq][sim_num] = []

# Lower lmax to that allowed by nside
args.lmax = min([3 * args.nside - 1, args.lmax])

if args.debug:
    import pylab as plt

    def debug_plot(map_file, name, sim_num, freq=None, ref_cls=None):
        m = hp.read_map(map_file, verbose=False)
        title = '%s, Sim %d' % (name, sim_num)
        if freq:
            title = '%s, %s GHz' % (title, freq)
        hp.mollview(m, title=title)
        plt.savefig(map_file.replace('.fits', '.png'), bbox_inches='tight')

        cls = hp.anafast(m, lmax=args.lmax)

        if ref_cls is None:
            plt.figure()
            plt.plot(cls, 'r', label='Sim %d' % sim_num)
            plt.xscale('log')
            plt.yscale('log')
            plt.title(title)
        else:
            cls_diff = (ref_cls[: args.lmax + 1] - cls) / ref_cls[: args.lmax + 1]
            fig, axs = plt.subplots(2, 1, sharex=True)
            axs[0].plot(ref_cls, 'k', label=name)
            axs[0].plot(cls, 'r', label='Sim %d' % sim_num)
            axs[0].legend()
            axs[0].set_xscale('log')
            axs[0].set_yscale('log')
            axs[0].set_title(title)
            axs[1].plot(cls_diff, 'r', label='Sim %d' % sim_num)
            axs[1].axhline(color='k')
            axs[1].set_ylim(-1, 1)
        plt.savefig(map_file.replace('.fits', '_spec.png'), bbox_inches='tight')

        plt.show()


# ===========================================================
# Set up random seed
# -----------------------------------------------------------
# maximum number of unique components per sim number
max_components = 100

# dictionary of offsets for each component that requires a
# unique realization of random numbers
comp_offsets = {
    'cmb': 0,
    'tsz': 1,
    'ksz': 2,
    'rg': 3,
    'dg-cl': 4,
    'dg-po': 5,
    'poisson_rg': 6,
    'poisson_dg': 7,
    'inst_noise_90': 8,
    'inst_noise_150': 9,
    'inst_noise_220': 10,
    'atm': 11,
    'gd': 12,
    'atm_uncorr_90': 13,
    'atm_uncorr_150': 14,
    'atm_uncorr_220': 15,
}


def set_seed(sim_num, comp):
    # random seed is determined uniquely for each sim index and
    # independent sim component
    np.random.seed(sim_num * max_components + comp_offsets[comp])


# ===========================================================
# Make simulated skies
# ===========================================================

# ===========================================================
# Make CMB realizations
# -----------------------------------------------------------
if args.cmb:
    core.log_notice('Creating CMB skies', unit='CMB')

    # Read CAMB spectrum
    # By default these units will be uK^2
    cls_dict = cmb.read_camb(args.camb_file, as_cls=True, lmax=args.lmax, lmin=0)
    if args.pol:
        camb_cls = np.asarray([cls_dict[x] for x in ['TT', 'EE', 'BB', 'TE']])
    else:
        camb_cls = cls_dict['TT']

    # convert camb_cls to G3units now
    camb_cls = np.asarray(camb_cls)
    camb_cls = camb_cls * (core.G3Units.uK ** 2.0)

    for sim_num in sim_range:
        core.log_info("Generating sim %d" % sim_num, unit='CMB')

        set_seed(sim_num, 'cmb')

        alms = hp.synalm(camb_cls, lmax=args.lmax, new=True, verbose=args.verbose)

        alm_file = get_filename('cmb', 'cmb_alms', sim_num)
        map_file = get_filename('cmb', 'cmb_map', sim_num)

        sim_tools.save_sims(
            alms=alms,
            store_alm=True,
            store_map=args.debug,
            alm_out=alm_file,
            map_out=map_file,
            lmax=args.lmax,
            nside=args.nside,
            pol=args.pol,
        )

        for freq in args.freqs:
            coadd_alms[freq][sim_num].append(alm_file)

        del alms

        if args.debug and sim_num == sim_range[0]:
            debug_plot(map_file, 'CMB', sim_num, ref_cls=cls_dict['TT'])

# ===========================================================
# Make foreground realizations
# -----------------------------------------------------------
if args.foregrounds:

    # -----------------------------------------------------------
    # Gaussian foregrounds
    # -----------------------------------------------------------
    for freq in args.freqs:
        core.log_notice(
            'Creating gaussian foregrounds for %sGHz channel' % freq, unit='Foregrounds'
        )
        band = '%sGHz' % (freq)
        fg_cls = foregrounds.get_foreground_sim_spectra(
            band,
            model=args.gaussian_fg_model,
            thermal_sz=args.gaussian_thermal_sz,
            kinetic_sz=args.gaussian_kinetic_sz,
            radio_galaxies=args.gaussian_radio_galaxies,
            dusty_galaxies=args.gaussian_dusty_galaxies,
            dg_clustering=args.gaussian_dg_clustering,
            gal_dust=args.galactic_dust,
            rescale=args.rescale,
            correct_spectra=args.correct_spectra,
            correction_file=args.correction_file,
            scaled_pspectra=args.scaled_pspectra,
            band0=args.band0,
            use_spt_datapoints=args.use_spt_datapoints,
            dg_clus_template_id=args.dg_clus_template_id,
            spec_index_rg=args.spec_index_rg,
            spec_index_dg_po=args.spec_index_dg_po,
            spec_index_dg_clus=args.spec_index_dg_clus,
            cib_temp=args.cib_temp * core.G3Units.kelvin,
        )

        for sim_num in sim_range:
            for key, cls in fg_cls.items():
                core.log_info(
                    "Generating %d GHz %s sim %d" % (freq, key, sim_num),
                    unit='Foregrounds',
                )

                # these foregrounds should be correlated between frequencies
                # so the seed is independent of freq
                set_seed(sim_num, key.lower())

                if args.pol:
                    # get polarization fractions
                    if 'sz' in key.lower():
                        pol_fraction = args.sz_pol_fraction
                    elif key.lower() == 'dg-cl':
                        # DG clustering term is not polarized
                        pol_fraction = 0
                    elif key.lower() == 'dg-po':
                        pol_fraction = args.dg_pol_fraction
                    elif 'rg' in key.lower():
                        pol_fraction = args.rg_pol_fraction
                    elif 'gd' in key.lower():
                        pol_fraction = 0
                    else:
                        raise KeyError(
                            'Cannot get pol_fraction for unrecognized %s' % key
                        )

                    # Assuming TE, TB, EB are all 0 for foregrounds
                    cls = [
                        cls,
                        cls * (pol_fraction ** 2) / 2,
                        cls * (pol_fraction ** 2) / 2,
                        cls * 0,
                    ]

                if 'gd' in key.lower():
                    # EE/BB = 2 for galactic dust
                    cls = [
                        cls[0] * 0,
                        cls[0] * 2,
                        cls[0],
                        cls[0] * 0,
                    ]

                alms = hp.synalm(cls, lmax=args.lmax, new=True, verbose=args.verbose)

                if not args.combine_gaussian_fg:
                    alm_file = get_filename(
                        'foregrounds', '%s_alms' % key.lower(), sim_num, freq
                    )
                    map_file = get_filename(
                        'foregrounds', '%s_map' % key.lower(), sim_num, freq
                    )

                    sim_tools.save_sims(
                        alms=alms,
                        store_alm=True,
                        store_map=args.debug,
                        alm_out=alm_file,
                        map_out=map_file,
                        lmax=args.lmax,
                        nside=args.nside,
                        pol=args.pol,
                    )

                    coadd_alms[freq][sim_num].append(alm_file)

                    del alms

                    if args.debug and sim_num == sim_range[0]:
                        debug_plot(
                            map_file,
                            'Foreground: %s' % key,
                            sim_num,
                            freq=freq,
                            ref_cls=cls,
                        )

                else:
                    try:
                        gauss_fg_alms += alms
                    except:
                        gauss_fg_alms = np.copy(alms)
                    del alms

            if args.combine_gaussian_fg:
                alm_file = get_filename(
                    'foregrounds', 'combined_gaussian_alms', sim_num, freq
                )

                map_file = get_filename(
                    'foregrounds', 'combined_gaussian_map', sim_num, freq
                )

                sim_tools.save_sims(
                    alms=gauss_fg_alms,
                    store_alm=True,
                    store_map=args.debug,
                    alm_out=alm_file,
                    map_out=map_file,
                    lmax=args.lmax,
                    nside=args.nside,
                    pol=args.pol,
                )

                coadd_alms[freq][sim_num].append(alm_file)

                del gauss_fg_alms

                if args.debug and sim_num == sim_range[0]:
                    debug_plot(
                        map_file,
                        'Foreground: %s' % key,
                        sim_num,
                        freq=freq,
                        ref_cls=cls,
                    )
        # cleanup
        fg_cls.clear()

    # -----------------------------------------------------------
    # Poisson foregrounds
    # -----------------------------------------------------------
    # Generate one realization per source population at 150GHz,
    # then scale fluxes appropriately.
    for comp in ['radio_galaxies', 'dusty_galaxies']:

        if not getattr(args, 'poisson_%s' % comp):
            continue

        core.log_notice(
            'Creating Poisson sims of %s' % comp.replace('_', ' '), unit='Foregrounds'
        )

        comp_short = 'rg' if comp == 'radio_galaxies' else 'dg'
        comp_spec = 'radio' if comp == 'radio_galaxies' else 'dust'

        pol_fraction = getattr(args, '%s_pol_fraction' % comp_short)

        spec_index = {
            90: getattr(args, 'spec_index_%s_90_150' % comp_spec),
            150: 1.0,
            220: getattr(args, 'spec_index_%s_220_150' % comp_spec),
        }

        fluxes, counts = foregrounds.get_poisson_source_counts(
            comp,
            band='150GHz',
            rg_model=args.poisson_rg_model,
            dg_model=args.poisson_dg_model,
            min_flux_limit=args.min_flux_limit,
            max_flux_limit=args.max_flux_limit,
        )

        for sim_num in sim_range:
            core.log_info(
                "Generating Poisson %s sim %d" % (comp_short.upper(), sim_num),
                unit='Foregrounds',
            )

            # foregrounds correlated between frequencies
            set_seed(sim_num, 'poisson_%s' % comp_short)

            maps = foregrounds.make_poisson_source_sim(
                fluxes,
                counts,
                comp,
                band='150GHz',
                pol=args.pol,
                pol_fraction=pol_fraction,
                nside=args.nside,
            )

            # Compute alms once here, and scale them for saving
            core.log_info("Converting %s poisson maps to alms..." % comp_spec)
            alms = np.asarray(hp.map2alm(maps, lmax=args.lmax, iter=0))
            core.log_info("...done converting %s poisson maps to alms." % comp_spec)

            for freq in args.freqs:
                scaling = (freq / 150.0) ** spec_index[freq]

                alm_file = get_filename(
                    'foregrounds', 'poisson_%s_alms' % comp_short, sim_num, freq
                )
                map_file = get_filename(
                    'foregrounds', 'poisson_%s_map' % comp_short, sim_num, freq
                )

                sim_tools.save_sims(
                    maps=maps * scaling,
                    alms=alms * scaling,
                    store_alm=True,
                    store_map=args.debug,
                    alm_out=alm_file,
                    map_out=map_file,
                    lmax=args.lmax,
                    nside=args.nside,
                    pol=args.pol,
                )

                coadd_alms[freq][sim_num].append(alm_file)

                if args.debug and sim_num == sim_range[0]:
                    debug_plot(
                        map_file,
                        'Foregrounds: Poisson %s' % comp_short.upper(),
                        sim_num,
                        freq=freq,
                    )

            # cleanup
            del maps, alms

    # -----------------------------------------------------------
    # Detected point sources and galaxy clusters
    # -----------------------------------------------------------
    # If you've decided to put in detected ones, poisson realizations
    # will not be needed. You should set poisson_radio_galaxies and
    # poisson_dusty_galaxies to False in the parameter yaml file.

    if getattr(args, 'detected_point_sources'):

        core.log_notice('Creating detected point sources')
        # most of the detected point sources are radio galaxies
        rg_pol_fraction = getattr(args, 'rg_pol_fraction')
        dg_pol_fraction = getattr(args, 'dg_pol_fraction')
        for freq in args.freqs:
            # sim_num always = 0, only one realization
            alm_file = get_filename(
                'foregrounds', 'detected_point_sources_alms', 0, freq
            )
            map_file = get_filename(
                'foregrounds', 'detected_point_sources_map', 0, freq
            )
            for sim_num in sim_range:
                coadd_alms[freq][sim_num].append(alm_file)
            if not os.path.isfile(alm_file):
                band = '%sGHz' % (freq)
                maps = foregrounds.make_detected_point_sources(
                    band=band,
                    pol=args.pol,
                    pol_fraction_dusty=dg_pol_fraction,
                    pol_fraction_radio=rg_pol_fraction,
                    nside=args.nside,
                    flux_range_150=[args.detected_min_flux, args.detected_max_flux],
                    ptsrc_file=args.ptsrc_file,
                )
                alms = np.asarray(hp.map2alm(maps, lmax=args.lmax, iter=0))
                sim_tools.save_sims(
                    maps=maps,
                    alms=alms,
                    store_alm=True,
                    store_map=args.debug,
                    alm_out=alm_file,
                    map_out=map_file,
                    lmax=args.lmax,
                    nside=args.nside,
                    pol=args.pol,
                )
                if args.debug:
                    debug_plot(
                        map_file,
                        'Foregrounds, detected point sources',
                        sim_num,
                        freq=freq,
                    )
                # cleanup
                del maps, alms

    if getattr(args, 'detected_spt_clusters'):

        core.log_notice('Creating detected spt3g clusters')
        for freq in args.freqs:
            # sim_num always = 0, only one realization
            alm_file = get_filename(
                'foregrounds', 'detected_spt_clusters_alms', 0, freq
            )
            map_file = get_filename('foregrounds', 'detected_spt_clusters_map', 0, freq)
            for sim_num in sim_range:
                coadd_alms[freq][sim_num].append(alm_file)
            if not os.path.isfile(alm_file):
                band = '%sGHz' % (freq)
                maps = foregrounds.make_spt_detected_clusters(
                    band=band,
                    pol=args.pol,
                    nside=args.nside,
                    sn_range=[
                        args.detected_cluster_min_sn,
                        args.detected_cluster_max_sn,
                    ],
                    cluster_file=args.cluster_file,
                )
                alms = np.asarray(hp.map2alm(maps, lmax=args.lmax, iter=0))
                sim_tools.save_sims(
                    maps=maps,
                    alms=alms,
                    store_alm=True,
                    store_map=args.debug,
                    alm_out=alm_file,
                    map_out=map_file,
                    lmax=args.lmax,
                    nside=args.nside,
                    pol=args.pol,
                )
                if args.debug:
                    debug_plot(
                        map_file,
                        'Foregrounds, detected spt3g clusters',
                        sim_num,
                        freq=freq,
                    )
                # cleanup
                del maps, alms


# ===========================================================
# Make noise realizations
# -----------------------------------------------------------
if args.noise:
    # pending: atmosphere T/Q/U are currently uncorrelated. must be fixed.
    noise_compnents = ['instrument', 'atmosphere_corr']
    if args.rho_oneoverf_noise != 1.0:
        noise_compnents.append('atmosphere_uncorr')
    noise_comp_short_dic = {
        'instrument': 'inst',
        'atmosphere_corr': 'atm_corr',
        'atmosphere_uncorr': 'atm_uncorr',
    }
    for freq in args.freqs:
        for comp in noise_compnents:
            core.log_notice(
                'Creating %s noise skies for %sGHz channel' % (comp, freq), unit='Noise'
            )

            # comp_short = 'inst' if comp == 'instrument' else 'atm'
            comp_short = noise_comp_short_dic[comp]

            nls = instrument.get_noise_sim_spectra(
                freq,
                component=comp,
                lmax=args.lmax,
                delta_t=getattr(args, 'delta_t_%d' % freq),
                lknee_t=getattr(args, 'lknee_t_%d' % freq),
                lknee_p=getattr(args, 'lknee_p_%d' % freq),
                alphaknee_t=getattr(args, 'alphaknee_t_%d' % freq),
                alphaknee_p=getattr(args, 'alphaknee_p_%d' % freq),
            )

            for sim_num in sim_range:
                core.log_info(
                    'Generating %d GHz %s sim %d' % (freq, comp, sim_num), unit='Noise'
                )

                if comp == 'instrument':
                    # independent noise realizations per frequency
                    set_seed(sim_num, 'inst_noise_%d' % freq)
                    nls_in = np.copy(nls)
                else:
                    # correlated 1/f noise between bands can be parameterised as
                    """
                    nl_nu1_nu1 = delta_t_nu1**2. * (1. + (elknee_nu1 * 1./el)**alphaknee_nu1 )
                    nl_nu1_nu2 = rho_oneoverf_noise * delta_t_nu1 * (elknee_nu1 * 1./el)**(alphaknee_nu1/2.)\
                                 * delta_t_nu2 * (elknee_nu1 * 1./el)**(alphaknee_nu1/2.)                    
                    """
                    # 1/fnoise correlated between frequencies
                    if comp == 'atmosphere_corr':
                        nls_in = np.copy(nls) * args.rho_oneoverf_noise
                        set_seed(sim_num, 'atm')
                    # 1/noise uncorrelated between frequencies
                    elif comp == 'atmosphere_uncorr':
                        nls_in = np.copy(nls) * (1.0 - args.rho_oneoverf_noise)
                        set_seed(sim_num, 'atm_uncorr_%d' % freq)
                # temperature noise
                alms = hp.synalm(
                    nls_in[0], lmax=args.lmax, new=True, verbose=args.verbose
                )
                if args.pol:
                    # pol noise
                    alms_q = hp.synalm(
                        nls_in[1], lmax=args.lmax, new=True, verbose=args.verbose
                    )
                    alms_u = hp.synalm(
                        nls_in[1], lmax=args.lmax, new=True, verbose=args.verbose
                    )
                    alms = np.asarray([alms, alms_q, alms_u])
                    del alms_q, alms_u

                if comp == 'instrument':
                    weight_map_file = eval('args.weight_map_%s' % freq)
                    if weight_map_file is not None:
                        for nn in range(len(alms)):
                            alms[nn] = sim_tools.create_inhomogeneous_noise(
                                alms[nn],
                                args.nside,
                                weight_map_file,
                                verbose=args.verbose,
                            )

                alm_file = get_filename('noise', '%s_alms' % comp_short, sim_num, freq)
                map_file = get_filename('noise', '%s_map' % comp_short, sim_num, freq)

                sim_tools.save_sims(
                    alms=alms,
                    store_alm=True,
                    store_map=args.debug,
                    alm_out=alm_file,
                    map_out=map_file,
                    lmax=args.lmax,
                    nside=args.nside,
                    pol=args.pol,
                )

                coadd_alms[freq][sim_num].append(alm_file)

                # cleanup
                del alms

                if args.debug and sim_num == sim_range[0]:
                    debug_plot(
                        map_file, '%s Noise' % comp.capitalize(), sim_num, freq=freq
                    )

# ===========================================================
# Combine components and save the final maps
# ===========================================================
core.log_notice('Combining sim components into maps', unit='Combined')
print('This is what we chose for beam: ', args.beam)
for freq in args.freqs:
    for sim_num in sim_range:
        print(coadd_alms[freq][sim_num])
        combined_map = sim_tools.combine_alms_into_map(
            coadd_alms[freq][sim_num],
            freq=freq,
            nside=args.nside,
            lmax=args.lmax,
            pol=args.pol,
            add_beam=args.beam,
            beamfile=args.beam_file,
            fwhm_90=args.fwhm_90,
            fwhm_150=args.fwhm_150,
            fwhm_220=args.fwhm_220,
            verbose=args.verbose,
        )
        filename = get_filename('total', 'total_map_3g', sim_num, freq)
        core.log_info('Saving spt3g map {}'.format(filename), unit='Combined')
        sim_tools.save_healpix_as_spt3g_map(
            combined_map, filename, maskfile=args.mask_file
        )

# storing the params in a timestamped file
args.alms = coadd_alms
tstamp = time.strftime('%Y%m%d_%H%M%S')
if args.config_file:
    prefix = os.path.splitext(os.path.basename(args.config_file))[0]
else:
    prefix = 'default_sims'
filename = os.path.join(args.output_root, '%s_%s.yaml' % (prefix, tstamp))
with open(filename, 'w') as f:
    yaml.dump(vars(args), f, default_flow_style=False)
