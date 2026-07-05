'''
Script to make CMB field maps.
'''
if __name__ != "__main__":
    raise ImportError

import argparse
import os, sys
import yaml
import numpy as np
from spt3g import core, cluster, std_processing, calibration, sources
from spt3g import timestreamflagging, todfilter, maps, pointing, transients

# Usage: master_field_mapmaker.py <input files.g3> -o outputmaps.g3
#            --config-file <config.yaml>

# =============================================================================
# Load in all settings for the map-making pipeline,
# either from command line or configuration yaml if one is given.
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Maps for a CMB field')
cli = parser.add_argument_group('Command Line Inputs', 'These settings are '
                                'specified via the command line only:')
cli.add_argument('input_files', nargs = '+',
                    help = ('The .g3 files containing observation data. The '
                            'offline-calibration.g3 file must be first.'))
cli.add_argument('-o', '--output', default = 'output.g3',
                    help = 'Output filename')
cli.add_argument('-z', '--compress', default=False, action='store_true',
                    help='Store output maps in compressed .g3.gz file')
cli.add_argument('--map-per-scan', action='store_true', help='Makes single scan maps.')
cli.add_argument('--produce-simstub', nargs='?', default=False, const=True,
                    help = ('Include this flag to produce a simstub. These '
                            'are used later for mock observations, and include'
                            ' information on flags and pointing. Same location'
                            ' as output, with "simstub-" prepended to file.'))
cli.add_argument('--simstub-only', action="store_true",
                 help="Create a simstub output file without a corresponding map file")
cli.add_argument('--sim', default = False, action = 'store_true',
                    help = ('Include this flag when mock-observing a '
                            'simulated map using simstubs as input files.'))
cli.add_argument('-m', '--sim-map',
                    help = 'Path to simulated .fits map to mock-observe.')
cli.add_argument('--no-weight-map', default=True, action='store_false', dest="store_weight_map",
                    help = ''.join(["If false, the output weight map will not be stored, ",
                            "though the output maps will *still be weighted*.", 
                            "It should be set to False if and only if you know", 
                            "what the weights are a priori (e.g. mock observing)."]))
cli.add_argument('--interp', default = False, action = 'store_true',
                    help = 'Interpolate the input sim map when mock-observing.')
cli.add_argument(
    "--no-error-on-zero",
    default=True,
    action="store_false",
    dest="error_on_zero",
    help="Do not raise an error on empty pixels in the input map when mock observing",
)
cli.add_argument('--add-ts-noise', default = False, action = 'store_true',
                    help = ('Add noise at the timestream level when '
                            'mock-observing.'))
cli.add_argument('--ts-noise-seed', default = -1, type=int,
                    help = ('Seed for random number generator when adding noise '
                            'at the timestream level in mock observations.  By '
                            'default, will seed based on current time.'))
cli.add_argument('-v', '--verbose', default=False, action = 'store_true',
                    help = 'Print frames to screen.')
cli.add_argument(
    "--map-center-ra",
    type=float,
    help="Flat sky map center in degrees. "
    "Map dec, width and height required if set. "
    "Command-line only, overrides --analysis map configuration.",
)
cli.add_argument(
    "--map-center-dec",
    type=float,
    help="Flat sky map center in degrees. "
    "Map ra, width and height required if set. "
    "Command-line only, overrides --analysis map configuration.",
)
cli.add_argument(
    "--map-width",
    type=float,
    help="Flat sky map width in degrees. "
    "Map ra, dec and height required if set. "
    "Command-line only, overrides --analysis map configuration.",
)
cli.add_argument(
    "--map-height",
    type=float,
    help="Flat sky map height in degrees. "
    "Map ra, dec and width required if set. "
    "Command-line only, overrides --analysis map configuration.",
)
cli.add_argument('--config-file',
                 help='.yaml file containing map-making parameters. '
                 'These will be treated as default values for the arguments below, '
                 'any of which may be overridden by supplying them at the command line. '
                 'See default_config.yaml for an example.')
config = parser.add_argument_group('Config File Inputs', 'These settings '
                                   'are specified in the config file:')
config.add_argument('--analysis', default='online',
                    help='Analysis tag to use for loading the appropriate '
                    'map stub and point source file')
config.add_argument('-s', '--map-source', default=None,
                    help = ('Name of observing field, used to configure map stub '
                            'and source lists for known fields.'))
config.add_argument('-r', '--map-resolution', default = 2.0, type=float,
                    help = 'Resolution of the output-map, in arcmin.')
config.add_argument('-p', '--map-projection', default = 5, type=int,
                    help = ('Projection of the output map. '
                            '0 = SansonFlamsteed (sinusoidal), '
                            '1 = Cartesian, '
                            '2 = Orthographic, '
                            '4 = Stereographic, '
                            '5 = LambertAzimuthalEqualArea, '
                            '9 = BICEP'))
config.add_argument('--temperature-only', default = False,
                    action = 'store_true',
                    help = 'Include this flag to make T-only maps.')
config.add_argument('--add-temperature-only', default = False, action = 'store_true',
                    help = 'Create a T-only map in addition to polarized')
config.add_argument('--healpix', nargs='?', default=False, const=True,
                    help='Output an healpix map in celestial coordinates.  '
                    '--map-center-ra and --map-center-dec flags are ignored when this option '
                    'is active.  The default polarization concenction is set to IAU.')
config.add_argument('--nside', default=2048, type=int,
                    help='Output healpix map Nside, default: 2048.  '
                    '--map-center-ra and --map-center-dec flags are ignored.  '
                    'Always produces a map in celestial coordinates')
config.add_argument('--flatsky', nargs='?', default=False, const=True,
                    help='Output a flat sky map. Can be supplied in conjunction'
                    'with the --healpix flag to run two map binners. If neither option'
                    'is specified default to a flat sky projection')
config.add_argument('--wafers-to-exclude', nargs = '+', default = [],
                    help = 'Wafers to exclude from mapmaking.')
config.add_argument('--wafers-to-include', nargs = '+', default=[],
                    help = ('Wafers to include in mapmaking. By default all '
                            'wafers present in BolometerProperties will be used.'))
config.add_argument('--bad-bolo-file', default = None,
                    help = ('Text file with list of bolos to NOT include in '
                            'mapmaking. Expects one bolo name per line.'))
config.add_argument('--bolo-file', default = None,
                    help = ('Text file with list of bolos to include in '
                            'mapmaking. Bolos not included in this list will be '
                            'excluded. Expects one bolo name per line.'))
config.add_argument('--include-pol-angles', nargs="+", type=float,
                    help="Include detectors with relative polarization angles in this set")
config.add_argument('--bands-to-use', nargs = '+', default=['90','150','220'],
                    choices=['90GHz', '150GHz', '220GHz', '90', '150', '220'],
                    help = ("Only use detectors in these observing bands. "
                            "Options are '90GHz', '150GHz', and/or '220GHz'"))
config.add_argument('--galaxy-mask', default=None, type=float, action='store',
                    help="If set, mask +/- the given latitude in degrees about the "
                    "Galactic plane when masking point sources for timestream "
                    "filtering and glitch finding.")
config.add_argument('--mask-point-sources', default = False,
                    action = 'store_true')
config.add_argument('--mask-resolution', default=0.25, type=float,
                    help="Map resolution to use for time domain masking, in arcmin")
config.add_argument('--point-source-file', default = None,
                    help="Point source file for making a point source mask."
                    "Must be readable using sources.read_point_source_mask_file(). "
                    "If not supplied, loaded using `sources.get_field_source_list()`"
                    "with the analysis tag set using the --analysis option.")
config.add_argument('--flux-threshold', default=None,
                    help='Point source flux threshold in mJy above which to '
                    'include in the mask')
config.add_argument('--cut-az-unwraps', default = False, action = 'store_true')
config.add_argument('--cut-dec-scan-speed', default=None, type=float, action='store',
                    help='Dec scan speed in deg/s above which to cut frames')
config.add_argument('--cut-az-glitches', default = False, action= 'store_true')
config.add_argument('--minnum-bolos-per-scan', type=int, default = 1)
config.add_argument('--apply-common-mode-filter', default = False,
                    action = 'store_true',
                    help = ('Subtract the average signal over a group of '
                            'detectors from each detector in that group. '
                            'If no further options are set, will use average '
                            'signal across all detectors. Options stack: e.g. '
                            '--cm-by-band used with --cm-by-squid will '
                            'construct groups of detectors that have the same '
                            'band and are on the same squid.'))
config.add_argument('--cm-by-wafer', default = False, action = 'store_true')
config.add_argument('--cm-by-band', default = False, action = 'store_true')
config.add_argument('--cm-by-squid', default = False, action = 'store_true')
config.add_argument('--cm-by-pol-angle', default=False, action='store_true')
config.add_argument('--mask-common-mode', default = False, action = 'store_true')
config.add_argument('--poly-order', default = 19, type = int,
                    help = ('Order of the polynomial filter applied to the '
                            'timestreams.'))
config.add_argument('--mask-poly-filter', default=None,
                    action=argparse.BooleanOptionalAction,
                    help="Override application of a source mask during poly filtering.")
config.add_argument('--filters-are-ell-based', default = False,
                    action = 'store_true',
                    help = ('If True, high-pass-cutoff and low-pass-cutoff '
                            'will be specified in ell-space. Otherwise Hz.'))
config.add_argument('--lls-method', default='quick', choices=['quick', 'qr', 'svd'],
                    help='LLS method to use for masked poly/HPF filter')
config.add_argument('--lls-iters', default=2, type=int,
                    help='Number of quick LLS iterations')
config.add_argument('--lls-joint-filter', action='store_true',
                    help='Apply polynomial and MHP filters simultaneously')
config.add_argument('--interpolate-fft-hpf', default = False, action = 'store_true',
                    help = ('Interpolate over bright sources and do the '
                    ' high-pass filter in Fourier space.'))
config.add_argument('--fft-hpf', default = False, action = 'store_true',
                    help = ('Do the high-pass filter in Fourier space but do not '
                            'interpolate over bright sources, for use when not '
                            'providing a point source file.'))
config.add_argument('--high-pass-cutoff', type = float, default = 300,
                    help = 'High-pass freq of filter applied to timestreams.')
config.add_argument('--low-pass-cutoff', type = float, default = 6600,
                    help = 'Low-pass freq of filter applied to timestreams.')
config.add_argument('--apply-notch-filter',
                    default = False, action = 'store_true',
                    help = ('Subtract best-fit (linear least squares based) '
                            'sine waves at those sample frequency values '
                            'computed by fftfreqs that fall within '
                            'user-specified frequency ranges. These ranges '
                            'should be recorded in a calibration frame '
                            'that is included in an input file.'))
config.add_argument('--notch-nyquist', default=False, action='store_true',
                        help=("Notch out the Nyquist frequency for fullrate data."))
config.add_argument('--unity-weights', action='store_true',
                    help='Sets detector weights to 1.0 when binning to a map.')
config.add_argument('--var-weights', default=False, action='store_true',
                    help='Used variance (rather than PDS) for computing timestream '
                    'weights, masked if --mask-point-sources is set.')
config.add_argument('--weights-are-ell-based', default = False,
                    action = 'store_true',
                    help = ('If True, weight-low-freq and lowweight-high-freq '
                            'will be specified in ell-space. Otherwise Hz.'))
config.add_argument('--weight-low-freq', type = float, default = 1.0,
                    help=("The lower limit of the frequency range, in Hz, used"
                          " to compute the weight of each detector's data."))
config.add_argument('--weight-high-freq', type = float, default = 4.0,
                    help=("The upper limit of the frequency range, in Hz, used"
                          " to compute the weight of each detector's data."))
config.add_argument('--pointing-model', default = 'online',
                    choices = ['online', 'offline','Online','Offline'])
config.add_argument('--nonlinear-pointing', action='store_true',
                    help='Use non-linear pointing model')
config.add_argument('--split-left-right', default = False,
                    help = ('Split left-going and right-going scans into '
                            'different maps. Can keep only left-going maps or '
                            'only right-going maps by passing "left" or '
                            '"right", respectively.'))
config.add_argument('--split-by-band', default = False, action = 'store_true',
                    help = ('Split data from different observing bands into '
                            'different maps.'))
config.add_argument('--split-by-wafer', default = False, action = 'store_true',
                    help = ('Split data from different wafers into '
                            'different maps.'))
config.add_argument('--split-by-pol-angle', default=False, action='store_true',
                    help='Split data with different polarization angles into '
                    'different maps')
config.add_argument('--split-by-chirality', default=False, action='store_true',
                    help='Split data with different antenna chirality into '
                    'different maps')
config.add_argument('--elnod-gain-matching', default = False,
                    action = 'store_true',
                    help = ('Use scan-by-scan elnod gain-matching method '
                            'which adjusts power in the pair-differenced '
                            'timestreams of the elnods. T stays same.'))
config.add_argument('--pd-weights', default = False,
                    action = 'store_true',
                    help = ('Weight using pair differenced timestreams.'))

config.add_argument('--min-cal-sn', default = 20.,
                    type = float, 
                    help = ('Min. calibrator SNR to identify bad detectors.'))
config.add_argument('--min-elnod-sn', default = 20.,
                    type = float, 
                    help = ('Min. el nod SNR to identify bad detectors.'))
config.add_argument('--fluxcal-sncut-90', default=None,
                    type = float, 
                    help = ('Minimum fluxcal SNR at 90 GHz to identify bad detectors.'))
config.add_argument('--fluxcal-sncut-150', default=None,
                    type = float, 
                    help = ('Minimum fluxcal SNR at 150 GHz to identify bad detectors.'))
config.add_argument('--fluxcal-sncut-220', default=None,
                    type = float, 
                    help = ('Minimum fluxcal SNR at 220 GHz to identify bad detectors.'))
config.add_argument('--no-glitch-finder', '--no-flag-glitches',
                    action='store_false', dest='flag_glitches',
                    help='Turns off the glitch finder.')
config.add_argument('--no-flag-oscillations',
                    action='store_false', dest='flag_oscillations',
                    help='Turns off the oscillating channel finder.')
config.add_argument('--glitch-flux-threshold', default=700,
                    help='Source flux threshold in mJy for masking bright sources '
                    'for the glitch finder')
g = config.add_mutually_exclusive_group()
g.add_argument('--deconvolve-tau', action='store_true',
               help='Deconvolve time constants and remove a time offset between '
               'pointing and detector timestreams.')
g.add_argument('--no-deconvolve-tau', default=False, action='store_false', dest='deconvolve_tau',
               help='Disable time constant deconvolution')
config.add_argument('--no-opacity-correction', default = False, action = 'store_true', 
                    help = ('Turns off the VFP sky opacity correction.'))
config.add_argument('--enforce-constant-scan-speed',
                    default = False, action = 'store_true',
                    help = ('Only use time samples of scans during which the telescope '
                            'scanned with a constant rate. The default arguments for '
                            'the algorithm are generally useful, but it is recommended '
                            'to examine these per field.'))
config.add_argument('--const-speed-frac-tol', default = 0.02, type = float, 
                    help = ('When enforcing constant scan speed, keep only those time '
                            'samples moving along the scan direction with speed within '
                            '(median_speed) +/- (median_speed * const_speed_frac_tol).'))
config.add_argument('--const-speed-rate-tol', default = 4, type = float,
                    help = ('When enforcing constant scan speed, the maximum speed in '
                            'arcsec per second to tolerate boresight movement in the '
                            'non-scan direction.'))
config.add_argument('--az-el', default = False, action = 'store_true',
                    help = ('Makes the output maps in az/el coordinates, not RA/DEC'))

args = argparse.Namespace()
argv = sys.argv[1:]

# If configuration yaml is specified, load it and pull default parameters from there.
if "--config-file" in argv:
    config_file = argv[argv.index("--config-file") + 1]
    with open(config_file, "r") as f:
        settings = yaml.safe_load(f)
    for k, v in settings.items():
        if k in ["map_center_ra", "map_center_dec", "map_width", "map_height"]:
            raise KeyError(
                "Your mapmaking config file includes arguments for defining the "
                "field map stub.  Remove these parameters and make sure that "
                "the field you're mapping is defined in CreateMasterFieldMapStub()."
            )
        if k == "map_source" and not v.strip():
            continue
        if k == "no_glitch_finder":
            core.log_warn(
                "no_glitch_finder key is deprecated, used flag_glitches instead."
            )
            k = "flag_glitches"
            v = not v
        setattr(args, k, v)

# update namespace with command-line parameters
args = parser.parse_args(argv, args)

# log input filenames for easier debugging
core.set_log_level(core.G3LogLevel.LOG_INFO, 'G3Reader')

# -----------------------------------------------------------------------------
# Reconfiguring some of the inputs:
# -----------------------------------------------------------------------------
if args.compress and not args.output.endswith('.gz'):
    args.output += '.gz'

if not args.filters_are_ell_based:
    hpf = core.G3Units.Hz * args.high_pass_cutoff
    lpf = core.G3Units.Hz * args.low_pass_cutoff
else:
    hpf = args.high_pass_cutoff
    lpf = args.low_pass_cutoff

if not args.weights_are_ell_based:
    wlf = core.G3Units.Hz * args.weight_low_freq
    whf = core.G3Units.Hz * args.weight_high_freq
else:
    wlf = args.weight_low_freq
    whf = args.weight_high_freq

bands = []
for band in args.bands_to_use:
    if 'ghz' not in str(band).lower():
        band=str(int(band))+'GHz'
    bands.append(band)
if args.sim and len(bands) > 1:
    raise RuntimeError("Only one band can be mock observed at one time")

bad_wafers = []
for wafer in args.wafers_to_exclude:
    if wafer not in [None, '']:
        bad_wafers.append(wafer.capitalize())

good_wafers = []
for wafer in args.wafers_to_include:
    if wafer not in [None, '']:
        good_wafers.append(wafer.capitalize())
        
bad_bolos = []
if args.bad_bolo_file is not None:
    bad_bolos = list(np.genfromtxt(args.bad_bolo_file, dtype=str))
        
good_bolos = []
if args.bolo_file is not None:
    good_bolos = list(np.genfromtxt(args.bolo_file, dtype=str))

if args.include_pol_angles:
    args.include_pol_angles = [x * core.G3Units.deg for x in args.include_pol_angles]

scan_directions = []
if args.split_left_right:
    if str(args.split_left_right).lower().startswith("l"):
        scan_directions = ["Left"]
    elif str(args.split_left_right).lower().startswith("right"):
        scan_directions = ["Right"]
    else:
        scan_directions = ["Left", "Right"]

pmodel = args.pointing_model.capitalize()
pointing_key = "{}{}Rotation".format(pmodel, "AzEl" if args.az_el else "RaDec")

# pointing checks units
cut_dec_scan_speed = None
if args.cut_dec_scan_speed is not None:
    cut_dec_scan_speed = args.cut_dec_scan_speed * core.G3Units.deg / core.G3Units.sec

const_speed_rate_tol = args.const_speed_rate_tol * core.G3Units.arcsec / core.G3Units.sec

# =============================================================================
# Before pipeline, get list of wafers and bolos to exclude if required
# -----------------------------------------------------------------------------
polang_list = args.include_pol_angles
if (
    args.map_source is None
    or (args.split_by_wafer and not len(good_wafers))
    or (args.split_by_pol_angle and not polang_list)
):
    for frame in cluster.GridFile(args.input_files):
        if frame.type == core.G3FrameType.Scan:
            break

        # discover field name
        if 'SourceName' in frame and not args.map_source:
            src = sources.get_field_season(frame["SourceName"])
            if src.startswith("spt3g-"):
                args.map_source = frame["SourceName"]

        # discover wafer list
        if 'BolometerProperties' in frame:
            if args.split_by_wafer and not len(good_wafers):
                calibration.SplitByWafer(wafers_key="Wafers")(frame)
                good_wafers = list(frame["Wafers"])

            if args.split_by_pol_angle:
                calibration.SplitByPolAngle(angles_key="PolAngles")(frame)
                polang_list = list(frame["PolAngles"])

        if (
            args.map_source is not None
            and (not args.split_by_wafer or len(good_wafers))
            and (not args.split_by_pol_angle or polang_list is not None)
        ):
            break

    if args.split_by_wafer and not len(good_wafers):
        raise ValueError("Unable to discover wafer list, --wafers-to-include required")
    args.wafers_to_include = good_wafers

    if args.split_by_pol_angle and polang_list is None:
        raise ValueError("Unable to discover pol angle list, missing calframe")

    if not args.map_source:
        raise ValueError("Unable to discover map source, --map-source required")

# -----------------------------------------------------------------------------
# Generate map stubs for use later in pipeline.
# -----------------------------------------------------------------------------

# Set map parameters for point souce masking
ps_mask = None
bright_ps_mask = None

# Load default source file if possible
if (
    args.mask_common_mode or
    args.mask_point_sources or
    args.interpolate_fft_hpf or
    (args.flag_glitches and not args.sim)
):
    if not args.point_source_file:
        args.point_source_file = sources.get_field_source_list(
            args.map_source, analysis=args.analysis,
        )

    from spt3g.std_processing import CreateMasterFieldMapStub

    # Use a sufficiently high-res mask for time domain filtering
    mask_params = CreateMasterFieldMapStub(
        args.map_source, res=args.mask_resolution * core.G3Units.arcmin
    )

    if args.galaxy_mask:
        gal_mask = maps.get_galactic_plane_mask(mask_params, args.galaxy_mask * core.G3Units.deg)

    if args.mask_common_mode or args.mask_point_sources or args.interpolate_fft_hpf:
        if args.flux_threshold is not None:
            args.flux_threshold *= core.G3Units.mJy

        # Fill map with the point source mask
        ps_mask = sources.make_point_source_map(
            mask_params,
            args.point_source_file,
            mask_above_threshold=args.flux_threshold,
        )
        if args.galaxy_mask:
            ps_mask |= gal_mask

    # Add second point source mask with only brightest sources for glitchfinder
    if args.flag_glitches and not args.sim:
        bright_ps_mask = sources.make_point_source_map(
            mask_params,
            args.point_source_file,
            mask_above_threshold=args.glitch_flux_threshold * core.G3Units.mJy,
        )
        if args.galaxy_mask:
            bright_ps_mask |= gal_mask

    if args.galaxy_mask:
        del gal_mask

else:
    args.point_source_file = None

# =============================================================================
# Begin pipeline
# -----------------------------------------------------------------------------
pipe = core.G3Pipeline()
pipe.Add(cluster.GridReader, filename=args.input_files)

stats = timestreamflagging.GenerateFlagStats(flag_key='Flags')

if not args.sim:

    new_ts_key = "RawTimestreams_I"

    # -------------------------------------------------------------------------
    # Drop certain data before further processing
    # -------------------------------------------------------------------------
    # Cut turnarounds, deduplicate metadata
    pipe.Add(std_processing.DropWasteFrames)
    pipe.Add(transients.balloons.BalloonAvoider)

    # Drop scans with common pointing issues
    pipe.Add(
        pointing.CheckBoresightPointing,
        scan_direction=scan_directions[0] if len(scan_directions) == 1 else None,
        cut_az_unwraps=args.cut_az_unwraps,
        cut_dec_scan_speed=cut_dec_scan_speed,
        cut_az_glitches=args.cut_az_glitches,
        enforce_constant_scan_speed=args.enforce_constant_scan_speed,
        const_speed_rate_tol=const_speed_rate_tol,
        const_speed_frac_tol=args.const_speed_frac_tol,
    )

    # -------------------------------------------------------------------------
    # Flag detectors that were not operated properly and/or had invalid data
    # -------------------------------------------------------------------------
    fluxcal_sncut = {}
    for b in [90, 150, 220]:
        v = getattr(args, f"fluxcal_sncut_{b}")
        if v is None or v < 0:
            continue
        fluxcal_sncut[b * core.G3Units.GHz] = v

    old_ts_key = new_ts_key
    new_ts_key = "Deflagged" + old_ts_key
    pipe.Add(
        timestreamflagging.PruneRawTimestreams,
        flag_key='Flags',
        input_ts_key=old_ts_key,
        output_ts_key=new_ts_key,
        q_ts_key="RawTimestreams_Q",
        bands=bands,
        exclude_wafers=bad_wafers or None,
        include_wafers=good_wafers or None,
        exclude_bolos=bad_bolos or None,
        include_bolos=good_bolos or None,
        include_pol_angles=args.include_pol_angles,
        flag_nan_offsets=True,
        flag_nan_bands=True,
        flag_nan_time_constants=args.deconvolve_tau,
        flag_incomplete_pairs=args.elnod_gain_matching or args.pd_weights,
        min_cal_sn=args.min_cal_sn,
        min_elnod_sn=args.min_elnod_sn,
        fluxcal_sncut=fluxcal_sncut or None,
        flag_stats=stats,
        destructive=True,
    )

    # -------------------------------------------------------------------------
    # Calibrate timestreams
    # -------------------------------------------------------------------------
    old_ts_key = new_ts_key
    new_ts_key = "CalTimestreams"
    pipe.Add(calibration.CalibrateRawTimestreams,
             i_data_key = old_ts_key, output = new_ts_key,
             opacity = not args.no_opacity_correction,
             elnod_gain_matching=args.elnod_gain_matching,
             keep_original=False)

    # -------------------------------------------------------------------------
    # Add pointing model
    # -------------------------------------------------------------------------
    pipe.Add(
        pointing.UpdateBoresightPointing,
        pointing_key=pointing_key,
        nonlinear=args.nonlinear_pointing,
    )

    # -------------------------------------------------------------------------
    # More flagging and removal of flagged detectors
    # -------------------------------------------------------------------------
    if args.notch_nyquist:
        old_ts_key = new_ts_key
        new_ts_key = 'NyquistNotchedTimestreams'
        pipe.Add(
            todfilter.notchfilter.NotchNyquist,
            ts_key=old_ts_key,
            out_key=new_ts_key,
        )
        pipe.Add(core.Delete, keys=[old_ts_key])

    old_ts_key = new_ts_key
    new_ts_key = "Deflagged" + old_ts_key
    pipe.Add(
        timestreamflagging.PruneCalTimestreams,
        flag_key='Flags',
        input_ts_key=old_ts_key,
        output_ts_key=new_ts_key,
        filter_mask_key='BrightFilterMask',
        ps_mask=bright_ps_mask,
        pointing_key=pointing_key,
        flag_glitches=args.flag_glitches,
        flag_oscillations=args.flag_oscillations,
        flag_stats=stats,
        min_bolos_per_scan=args.minnum_bolos_per_scan,
        destructive=True,
    )

else:
    # Load the simulated map
    if args.sim_map.endswith('.fits'):
        m = maps.fitsio.load_skymap_fits(cluster.get_grid_file(args.sim_map))
    elif args.sim_map.endswith('.g3') or args.sim_map.endswith('.g3.gz'):
        for frame in cluster.GridFile(args.sim_map):
            if frame.type == core.G3FrameType.Map:
                m = frame
                break
        else:
            raise IOError("No map found in input file %s" % args.sim_map)
    else:
        raise ValueError("Unrecognized file type for input %s" % args.sim_map)

    new_ts_key = "CalTimestreams"
    pipe.Add(
        todfilter.CreateSimTimestreams,
        output_key=new_ts_key,
        sim_map=m["T"] if args.temperature_only else m,
        band=calibration.band_to_value(bands[0]),
        pointing_key=pointing_key,
        interp=args.interp,
        error_on_zero=args.error_on_zero,
        add_white_noise=args.add_ts_noise,
        noise_seed=args.ts_noise_seed,
        bad_bolos=bad_bolos,
        deconvolve_tau=args.deconvolve_tau,
        destructive=True,
    )

# -----------------------------------------------------------------------------
# Timestream filtering
# -----------------------------------------------------------------------------

old_ts_key = new_ts_key
new_ts_key = 'FilteredTimestreams'
pipe.Add(todfilter.TodFiltering,
         # filtering options
         poly_order = args.poly_order,
         poly_masked=args.mask_poly_filter,
         filters_are_ell_based = args.filters_are_ell_based,
         hpf_filter_frequency=hpf,
         fft_hpf=args.fft_hpf,
         interpolate_fft_hpf=args.interpolate_fft_hpf,
         lpf_filter_frequency = lpf,
         filter_mask_key="FilterMask" if ps_mask is not None else None,
         common_mode=args.apply_common_mode_filter,
         cm_masked=args.mask_common_mode,
         cm_by_band=args.cm_by_band,
         cm_by_wafer=args.cm_by_wafer,
         cm_by_squid=args.cm_by_squid,
         cm_by_pol_angle=args.cm_by_pol_angle,
         notch_filter=args.apply_notch_filter,
         notch_filter_masked=args.mask_point_sources,
         # boiler plate
         ts_in_key = old_ts_key,
         ts_out_key = new_ts_key,
         delete_input_ts = True,
         use_dynamic_source_filter = False,
         point_source_mask=ps_mask,
         point_source_pointing_store_key=pointing_key,
         boresight_ra_key = pmodel+'BoresightRa',
         boresight_dec_key = pmodel+'BoresightDec',
         ell_filter_effective_sample_rate_key = 'EllFilterSampleRate',
         fft_padding_key = 'FftPadding',
         filter_key = 'FourierFilter',
         time_constants_key='TimeConstants' if args.deconvolve_tau else None,
         time_offset_key='TimeOffset' if args.deconvolve_tau else None,
         lls_method=args.lls_method,
         lls_iters=args.lls_iters,
         lls_joint_filter=args.lls_joint_filter,
)

if not args.sim:
    # -------------------------------------------------------------------------
    # Calculate weights, flag bolos with bad weights
    # -------------------------------------------------------------------------
    old_ts_key = new_ts_key
    new_ts_key = 'Deflagged' + old_ts_key
    pipe.Add(
        todfilter.weighting.AddTimestreamWeights,
        input_timestreams=old_ts_key,
        output_timestreams=new_ts_key,
        output_weights="TodWeights",
        pairdiff=args.pd_weights,
        var=args.var_weights,
        mask_key="FilterMask" if args.mask_point_sources else None,
        ell_based=args.weights_are_ell_based,
        pointing_model=pmodel,
        low_freq=wlf,
        high_freq=whf,
        sigmaclip_thresh=3.0,
        flag_key="Flags",
        destructive=True,
    )

# -----------------------------------------------------------------------------
# Generate map meta-data to access later
# -----------------------------------------------------------------------------
pipe.Add(std_processing.AddMetaData,
         final_ts_key=new_ts_key,
         stats = stats,
         args = vars(args))

# -----------------------------------------------------------------------------
# Bin the timestreams into maps
# -----------------------------------------------------------------------------
if not args.simstub_only:
    if not args.flatsky and not args.healpix:
        args.flatsky = True

    extra_map_params = {}
    if args.map_center_ra is not None and args.map_center_dec is not None:
        extra_map_params.update(
            ra_center=args.map_center_ra * core.G3Units.deg,
            dec_center=args.map_center_dec * core.G3Units.deg,
            width=args.map_width * core.G3Units.deg,
            height=args.map_height * core.G3Units.deg,
        )

    pipe.Add(
        std_processing.mapmaking.SplitMapBinner,
        timestreams=new_ts_key,
        scan_directions=scan_directions if args.split_left_right else None,
        bands=bands if args.split_by_band else None,
        wafers=good_wafers if args.split_by_wafer else None,
        pol_angles=polang_list if args.split_by_pol_angle else None,
        chirals=["ChiralA", "ChiralB"] if args.split_by_chirality else None,
        flatsky_stub=args.flatsky,
        healpix_stub=args.healpix,
        field=args.map_source,
        analysis=args.analysis,
        res=args.map_resolution * core.G3Units.arcmin,
        proj=args.map_projection,
        pol=not args.temperature_only,
        nside=args.nside,
        pointing=pointing_key,
        detector_weights="" if args.unity_weights else "TodWeights",
        store_weight_map=args.store_weight_map,
        map_per_scan=args.map_per_scan,
        **extra_map_params,
    )

if args.verbose:
    pipe.Add(std_processing.mapmaking.DumpScans)

# -----------------------------------------------------------------------------
# Write simstub
# -----------------------------------------------------------------------------
if args.produce_simstub or args.simstub_only:
    if isinstance(args.produce_simstub, str):
        simstub_output = args.produce_simstub
    else:
        outbase = args.output.split('/')[-1]
        simstub_output = args.output.replace(outbase, 'simstub_' + outbase)
    if args.compress and not simstub_output.endswith(".gz"):
        simstub_output += ".gz"

    pipe.Add(std_processing.mapmaking.SimStubWriter, filename=simstub_output)

# -----------------------------------------------------------------------------
# Drop Scan frames
# -----------------------------------------------------------------------------
pipe.Add(lambda fr: fr.type != core.G3FrameType.Scan)

# -----------------------------------------------------------------------------
# Write to file
# -----------------------------------------------------------------------------
if not args.simstub_only and args.healpix:
    if isinstance(args.healpix, str):
        healpix_output = args.healpix
    elif args.flatsky:
        outbase = args.output.split('/')[-1]
        healpix_output = args.output.replace(outbase, 'healpix_' + outbase)
    else:
        healpix_output = args.output
    if args.compress and not healpix_output.endswith(".gz"):
        healpix_output += ".gz"
    core.log_notice("Writing healpix outputs to {}".format(healpix_output))

    pipe.Add(
        std_processing.mapmaking.MapWriter,
        map_type=maps.HealpixSkyMap,
        filename=healpix_output,
        add_tonly=not args.temperature_only and args.add_temperature_only,
    )
if not args.simstub_only and args.flatsky:
    flatsky_output = args.flatsky if isinstance(args.flatsky, str) else args.output
    if args.compress and not flatsky_output.endswith('.gz'):
        flatsky_output += '.gz'
    core.log_notice("Writing flatsky outputs to {}".format(flatsky_output))

    pipe.Add(
        std_processing.mapmaking.MapWriter,
        map_type=maps.FlatSkyMap,
        filename=flatsky_output,
        add_tonly=not args.temperature_only and args.add_temperature_only,
    )

# =============================================================================
# Run the pipeline
# -----------------------------------------------------------------------------
pipe.Run(profile=True)
