from astropy.io import fits
from astropy.wcs import WCS
import sys
sys.path.append('/home/grespanm/github/TEGLIE/teglie_scripts/')
import settings
#from utils import from_fits_to_array
import numpy as np
import os
import tracemalloc
import pandas as pd
import matplotlib as mpl
import io
from skimage.transform import resize
import cv2
from astronomaly.base.base_dataset import Dataset
from astronomaly.base import logging_tools
from astronomaly.utils import utils
import make_rgb
mpl.use('Agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas  # noqa: E402, E501
import matplotlib.pyplot as plt  # noqa: E402
import pickle
import h5py

def change_specialcharacters(string):
    return string.replace('KIDS', 'KiDS_DR4.0').replace('.', 'p').replace('-', 'm').rstrip(' ')


def convert_array_to_image(arr, plot_cmap='viridis', interpolation='bicubic'):
    """
    Function to convert an array to a png image ready to be served on a web
    page.

    Parameters
    ----------
    arr : np.ndarray
        Input image
    plot_cmap : str, optional
        Which colourmap to use
    interpolation : str, optional
        Allows interpolation so low res images don't look so blocky

    Returns
    -------
    png image object
        Object ready to be passed directly to the frontend
    """
    with mpl.rc_context({'backend': 'Agg'}):
        #print(arr.shape)
        #if len(arr.shape)==2:
            # if greyscale plot just that
        fig = plt.figure(figsize=(1, 1), dpi=4 * arr.shape[1])
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        plt.imshow(arr, cmap=plot_cmap, origin='lower',
                interpolation=interpolation)
        output = io.BytesIO()
        FigureCanvas(fig).print_png(output)
        plt.close(fig)
        '''
        else:
            #if 3 bands plot r band and rgb
            fig = plt.figure(figsize=(1, 1), dpi=6 * arr.shape[1])
            
            
            #code to have more than one image shown
            #ax1 = fig.add_subplot(1, 2, 1)
            #r band #np.rot90(arr[:,:,0]),
            #ax1.imshow(arr[:,:,0], 
            #            cmap=plot_cmap, origin='lower', interpolation=interpolation)
            #ax1.set_axis_off()
            #ax2 = fig.add_subplot(1, 2, 2)
            #rgb
            #ax2.imshow(make_rgb.make_rgb_one_image(arr, return_img=True),
            #            cmap=plot_cmap, origin='lower', interpolation=interpolation)
            #ax2.set_axis_off()
            
            ## code ot use if you load more channels:
            #plt.imshow(make_rgb.make_rgb_one_image(arr, return_img=True),
            #            origin='lower', interpolation=interpolation)

            plt.imshow(arr)
            print(arr)
            plt.axis('off')
            output = io.BytesIO()
            FigureCanvas(fig).print_png(output)
            plt.close(fig)
        '''        

    


    return output


def apply_transform(cutout, transform_function):
    """
    Applies the transform function(s) given at initialisation to the image.

    Parameters
    ----------
    cutout : np.ndarray
        Cutout of image

    Returns
    -------
    np.ndarray
        Transformed cutout
    """
    if transform_function is not None:
        try:
            len(transform_function)
            new_cutout = cutout
            for f in transform_function:
                new_cutout = f(new_cutout)
            cutout = new_cutout
        except TypeError:  # Simple way to test if there's only one function
            cutout = transform_function(cutout)
    #print(np.shape(cutout))
    return cutout


class AstroImage:
    def __init__(self, filenames, file_type='fits', fits_index=None, name=''):
        """
        Lightweight wrapper for an astronomy image from a fits file

        Parameters
        ----------
        filenames : list of files
            Filename of fits file to be read. Can be length one if there's only
            one file or multiple if there are multiband images
        fits_index : integer
            Which HDU object in the list to work with

        """
        print('Reading image data from %s...' % filenames[0])
        self.filenames = filenames
        self.file_type = file_type
        self.metadata = {}
        self.wcs = None
        self.fits_index = fits_index
        self.hdul_list = []

        try:
            for f in filenames:
                hdul = fits.open(f, memmap=True)
                self.hdul_list.append(hdul)

        except FileNotFoundError:
            raise FileNotFoundError("File", f, "not found")

        # get a test sample
        self.get_image_data(0, 10, 0, 10)

        if len(name) == 0:
            self.name = self._strip_filename()
        else:
            self.name = name

        print('Done!')

    def get_image_data(self, row_start, row_end, col_start, col_end):
        """Returns the image data from a fits HDUlist object

        Parameters
        ----------
        Returns
        -------
        np.array
            Image data
        """
        images = []
        rs = row_start
        re = row_end
        cs = col_start
        ce = col_end

        for hdul in self.hdul_list:
            if self.fits_index is None:
                for i in range(len(hdul)):
                    self.fits_index = i
                    # snap1 = tracemalloc.take_snapshot()
                    dat = hdul[self.fits_index].data
                    # snap2 = tracemalloc.take_snapshot()
                    # diff = snap2.compare_to(snap1, 'lineno')
                    # print(diff[0].size_diff)
                    if dat is not None:
                        if len(dat.shape) > 2:
                            dat = dat[0][0]
                        image = dat[rs:re, cs:ce]
                        break
            else:
                dat = hdul[self.fits_index].data
                if len(dat.shape) > 2:
                    dat = dat[0][0]
                image = dat[rs:re, cs:ce]

            self.metadata = dict(hdul[self.fits_index].header)
            if self.wcs is None:
                self.wcs = WCS(hdul[self.fits_index].header, naxis=2)

            if len(image.shape) > 2 and image.shape[-1] > 3:
                image = image[:, :, 0]
            if len(image.shape) > 2:
                image = np.squeeze(image)
            images.append(image)

        if len(images) > 1:
            # Should now be a 3d array with multiple channels
            image = np.dstack(images)
            self.metadata['NAXIS3'] = image.shape[-1]
        else:
            image = images[0]  # Was just the one image

        return image

    def get_image_shape(self):
        """
        Efficiently returns the shape of the image.

        Returns
        -------
        tuple
            Image shape
        """
        return (self.metadata['NAXIS1'], self.metadata['NAXIS2'])

    def clean_up(self):
        """
        Closes all open fits files so they don't remain in memory.
        """
        print("Closing Fits files...")
        for hdul in self.hdul_list:
            hdul.close()
        logging_tools.log("Fits files closed successfully.")
        print("Files closed.")

    def _strip_filename(self):
        """
        Tiny utility function to make a nice formatted version of the image
        name from the input filename string

        Returns
        -------
        string
            Formatted file name

        """
        s1 = self.filenames[0].split(os.path.sep)[-1]
        # extension = s1.split('.')[-1]
        return s1

    def get_coords(self, x, y):
        """
        Returns the RA and DEC coordinates for a given set of pixels.

        Parameters
        ----------
        x : int
            x pixel value
        y : y
            y pixel value

        Returns
        -------
        ra, dec
            Sky coordinates
        """

        return self.wcs.wcs_pix2world(x, y, 0)


class ImageDataset(Dataset):
    def __init__(self, fits_index=None, window_size=101, window_shift=None,
                 display_image_size=None, adaptive_sizing=False,
                 min_window_size=100,
                 band_prefixes=[], bands_rgb={},
                 transform_function=None, display_transform_function=None,
                 plot_square=False, catalogue=None,
                 plot_cmap='hot', 
                 display_interpolation=None,
                 **kwargs):
        """
        Read in a set of images either from a directory or from a list of file
        paths (absolute). Inherits from Dataset class.

        Parameters
        ----------
        filename : str
            If a single file (of any time) is to be read from, the path can be
            given using this kwarg. 
        directory : str
            A directory can be given instead of an explicit list of files. The
            child class will load all appropriate files in this directory.
        list_of_files : list
            Instead of the above, a list of files to be loaded can be
            explicitly given.
        output_dir : str
            The directory to save the log file and all outputs to. Defaults to
            './' 
        fits_index : integer, optional
            If these are fits files, specifies which HDU object in the list to
            work with
        window_size : int, tuple or list, optional
            The size of the cutout in pixels. If an integer is provided, the 
            cutouts will be square. Otherwise a list of 
            [window_size_x, window_size_y] is expected.
        window_shift : int, tuple or list, optional
            The size of the window shift in pixels. If the shift is less than 
            the window size, a sliding window is used to create cutouts. This 
            can be particularly useful for (for example) creating a training 
            set for an autoencoder. If an integer is provided, the shift will 
            be the same in both directions. Otherwise a list of
            [window_shift_x, window_shift_y] is expected.
        display_image_size : None or int
            The size of the image to be displayed on the
            web page. If the image is smaller than this, it will be
            interpolated up to the higher number of pixels. If larger, it will
            be downsampled. If None, will default to the existing width of the
            image.
        display_interpolation : None or str
            Passed to matplotlib imshow. If you are displaying images at a 
            similar size to their native resolution interpolation isn't usually
            necessary. But if they are being upsampled with a larger 
            display_image_size, 'bicubic' will allow for a smoother looking 
            image.
        adaptive_sizing : Boolean, optional
            Allows the size of the window to be adapted based on the size of 
            the object. Requires an "obj_size" column in the catalogue. This 
            should be the desired window width in pixels. Note that a minimum
            window size can also be provided to provide a lower limit for the
            adaptive sizing. We currently only support a single number for
            the size so all cutouts will be square. The different sized images 
            will be resized into window_size.
        min_window_size : int, optional
            When used with adapative_sizing, this provides a lower limit for
            the window size (to avoid tiny sources made of very few pixels 
            being blown up). Default is 128 pixels, set to 0 to allow all sizes
        band_prefixes : list
            Allows you to specify a prefix for an image which corresponds to a
            band identifier. This has to be a prefix and the rest of the image
            name must be identical in order for Astronomaly to detect these
            images should be stacked together. 
        bands_rgb : Dictionary
            Maps the input bands (in separate folders) to rgb values to allow
            false colour image plotting. Note that here you can only select
            three bands to plot although you can use as many bands as you like
            in band_prefixes. The dictionary should have 'r', 'g' and 'b' as
            keys with the band prefixes as values.
        transform_function : function or list, optional
            The transformation function or list of functions that will be 
            applied to each cutout. The function should take an input 2d array 
            (the cutout) and return an output 2d array. If a list is provided, 
            each function is applied in the order of the list.
        catalogue : pandas.DataFrame or similar
            A catalogue of the positions of sources around which cutouts will
            be extracted. Note that a cutout of size "window_size" will be
            extracted around these positions and must be the same for all
            sources. 
        plot_square : bool, optional
            If True this will add a white border indicating the boundaries of
            the original cutout when the image is displayed in the webapp.
        plot_cmap : str, optional
            The colormap with which to plot the image
        interpolation : str, optional
            Allows interpolation in the display image so low res images don't
            look so blocky
        """

        super().__init__(fits_index=fits_index, window_size=window_size,
                         window_shift=window_shift,
                         display_image_size=display_image_size,
                         adaptive_sizing=adaptive_sizing,
                         min_window_size=min_window_size,
                         band_prefixes=band_prefixes, bands_rgb=bands_rgb,
                         transform_function=transform_function,
                         display_transform_function=display_transform_function,
                         plot_square=plot_square, catalogue=catalogue,
                         plot_cmap=plot_cmap,
                         display_interpolation=display_interpolation,
                         **kwargs)
        self.known_file_types = ['fits', 'fits.fz', 'fits.gz',
                                 'FITS', 'FITS.fz', 'FITS.gz']
        self.data_type = 'image'

        images = {}
        tracemalloc.start()

        if len(band_prefixes) != 0:
            # Get the matching images in different bands
            bands_files = {}

            for p in band_prefixes:
                for f in self.files:
                    if p in f:
                        start_ind = f.find(p)
                        end_ind = start_ind + len(p)
                        flname = f[end_ind:]
                        if flname not in bands_files.keys():
                            bands_files[flname] = [f]
                        else:
                            bands_files[flname] += [f]

            for k in bands_files.keys():
                extension = k.split('.')[-1]
                # print(k, extension)
                if extension == 'fz' or extension == 'gz':
                    extension = '.'.join(k.split('.')[-2:])
                if extension in self.known_file_types:
                    try:
                        astro_img = AstroImage(bands_files[k],
                                               file_type=extension,
                                               fits_index=fits_index,
                                               name=k)
                        images[k] = astro_img

                    except Exception as e:
                        msg = "Cannot read image " + k + "\n \
                            Exception is: " + (str)(e)
                        logging_tools.log(msg, level="ERROR")

            # Also convert the rgb dictionary into an index dictionary
            # corresponding
            if len(bands_rgb) == 0:
                self.bands_rgb = {'r': 0, 'g': 1, 'b': 2}
            else:
                self.bands_rgb = {}
                for k in bands_rgb.keys():
                    band = bands_rgb[k]
                    ind = band_prefixes.index(band)
                    self.bands_rgb[k] = ind
        else:
            for f in self.files:
                extension = f.split('.')[-1]
                if extension == 'fz' or extension == 'gz':
                    extension = '.'.join(f.split('.')[-2:])
                if extension in self.known_file_types:
                    try:
                        astro_img = AstroImage([f],
                                               file_type=extension,
                                               fits_index=fits_index)
                        images[astro_img.name] = astro_img
                    except Exception as e:
                        msg = "Cannot read image " + f + "\n \
                            Exception is: " + (str)(e)
                        logging_tools.log(msg, level="ERROR")
        if len(list(images.keys())) == 0:
            msg = "No images found, Astronomaly cannot proceed."
            logging_tools.log(msg, level="ERROR")
            raise IOError(msg)

        try:
            self.window_size_x = window_size[0]
            self.window_size_y = window_size[1]
        except TypeError:
            self.window_size_x = window_size
            self.window_size_y = window_size

        # Allows sliding windows
        if window_shift is not None:
            try:
                self.window_shift_x = window_shift[0]
                self.window_shift_y = window_shift[1]
            except TypeError:
                self.window_shift_x = window_shift
                self.window_shift_y = window_shift
        else:
            self.window_shift_x = self.window_size_x
            self.window_shift_y = self.window_size_y

        if adaptive_sizing:
            err = False
            if catalogue is None:
                err = True
            else:
                if 'obj_size' not in catalogue.columns:
                    err = True
            if err:
                err_msg = ("To use adaptive sizing, a catalogue must be "
                           "provided with a 'obj_size' column.")
                logging_tools.log(err_msg, level='ERROR')
                raise(ValueError(err_msg))

        self.adaptive_sizing = adaptive_sizing
        self.min_window_size = min_window_size

        self.images = images
        self.transform_function = transform_function
        if display_transform_function is None:
            self.display_transform_function = transform_function
        else:
            self.display_transform_function = display_transform_function

        self.plot_square = plot_square
        self.plot_cmap = plot_cmap
        self.display_interpolation = display_interpolation
        self.catalogue = catalogue
        self.display_image_size = display_image_size
        self.band_prefixes = band_prefixes

        self.metadata = pd.DataFrame(data=[])
        if self.catalogue is None:
            self.create_catalogue()
        else:
            self.convert_catalogue_to_metadata()
            print('A catalogue of ', len(self.metadata),
                  'sources has been provided.')

        if 'original_image' in self.metadata.columns:
            for img in np.unique(self.metadata.original_image):
                if img not in images.keys():
                    logging_tools.log('Image ' + img + """ found in catalogue 
                        but not in provided image data. Removing from 
                        catalogue.""", level='WARNING')
                    msk = self.metadata.original_image == img
                    self.metadata.drop(self.metadata.index[msk], inplace=True)
                    print('Catalogue reduced to ', len(self.metadata),
                          'sources')

        self.index = self.metadata.index.values

    def create_catalogue(self):
        """
        If a catalogue is not supplied, this will generate one by cutting up
        the image into cutouts.
        """
        print('No catalogue found, one will automatically be generated by \
               splitting the image into cutouts governed by the window_size..')
        for image_name in list(self.images.keys()):
            astro_img = self.images[image_name]
            img_shape = astro_img.get_image_shape()

            # Remember, numpy array index of [row, column]
            # corresponds to [y, x]
            xvals = np.arange(self.window_size_x // 2,
                              img_shape[1] - self.window_size_x // 2,
                              self.window_shift_x)
            yvals = np.arange(self.window_size_y // 2,
                              img_shape[0] - self.window_size_y // 2,
                              self.window_shift_y)
            X, Y = np.meshgrid(xvals, yvals)

            x_coords = X.ravel()
            y_coords = Y.ravel()
            ra, dec = astro_img.get_coords(x_coords, y_coords)
            original_image_names = [image_name] * len(x_coords)

            new_df = pd.DataFrame(data={
                'original_image': original_image_names,
                'x': x_coords,
                'y': y_coords,
                'ra': ra,
                'dec': dec,
                'peak_flux': [-1] * len(ra)})
            self.metadata = pd.concat((self.metadata, new_df),
                                      ignore_index=True)
        self.metadata.index = self.metadata.index.astype('str')
        print('A catalogue of ', len(self.metadata), 'cutouts has been \
               created.')
        print('Done!')

    def convert_catalogue_to_metadata(self):

        if 'original_image' not in self.catalogue.columns:
            if len(self.images) > 1:
                logging_tools.log("""If multiple fits images are used the
                                  original_image column must be provided in
                                  the catalogue to identify which image the 
                                  source belongs to.""",
                                  level='ERROR')

                raise ValueError("Incorrect input supplied")

            else:
                self.catalogue['original_image'] = \
                    [list(self.images.keys())[0]] * len(self.catalogue)

        if 'objid' not in self.catalogue.columns:
            self.catalogue['objid'] = np.arange(len(self.catalogue))

        if 'peak_flux' not in self.catalogue.columns:
            self.catalogue['peak_flux'] = [np.NaN] * len(self.catalogue)

        cols = ['original_image', 'x', 'y']

        for c in cols[1:]:
            if c not in self.catalogue.columns:
                logging_tools.log("""If a catalogue is provided the x and y
                columns (corresponding to pixel values) must be present""",
                                  level='ERROR')

                raise ValueError("Incorrect input supplied")

        if 'ra' in self.catalogue.columns:
            cols.append('ra')
        if 'dec' in self.catalogue.columns:
            cols.append('dec')
        if 'peak_flux' in self.catalogue.columns:
            cols.append('peak_flux')
        if 'obj_size' in self.catalogue.columns:
            cols.append('obj_size')

        met = {}
        for c in cols:
            met[c] = self.catalogue[c].values

        the_index = np.array(self.catalogue['objid'].values, dtype='str')
        self.metadata = pd.DataFrame(met, index=the_index)
        self.metadata['x'] = self.metadata['x'].astype('int')
        self.metadata['y'] = self.metadata['y'].astype('int')

    def get_sample(self, idx):
        """
        Returns the data for a single sample in the dataset as indexed by idx.

        Parameters
        ----------
        idx : string
            Index of sample

        Returns
        -------
        nd.array
            Array of image cutout
        """

        x0 = self.metadata.loc[idx, 'x']
        y0 = self.metadata.loc[idx, 'y']
        original_image = self.metadata.loc[idx, 'original_image']
        this_image = self.images[original_image]

        if self.adaptive_sizing:
            wid = self.metadata.loc[idx, 'obj_size']
            if wid < self.min_window_size:
                wid = self.min_window_size
            x_wid = wid // 2
            y_wid = wid // 2
        else:
            x_wid = self.window_size_x // 2
            y_wid = self.window_size_y // 2

        y_start = y0 - y_wid
        y_end = y0 + y_wid
        x_start = x0 - x_wid
        x_end = x0 + x_wid

        invalid_y = y_start < 0 or y_end > this_image.metadata['NAXIS1']
        invalid_x = x_start < 0 or x_end > this_image.metadata['NAXIS2']
        if invalid_y or invalid_x:
            naxis3_present = 'NAXIS3' in this_image.metadata.keys()
            if naxis3_present and this_image.metadata['NAXIS3'] > 1:
                shp = [self.window_size_y,
                       self.window_size_x,
                       this_image.metadata['NAXIS3']]
            else:
                shp = [self.window_size_y, self.window_size_x]
            cutout = np.ones((shp)) * np.nan
        else:
            cutout = this_image.get_image_data(y_start, y_end, x_start, x_end)

        if self.adaptive_sizing:
            new_shape = (self.window_size_x, self.window_size_y)
            cutout = resize(cutout, new_shape, anti_aliasing=False)

        if self.metadata.loc[idx, 'peak_flux'] == -1:
            if np.any(np.isnan(cutout)):
                flx = -1
            else:
                flx = np.max(cutout)
            self.metadata.loc[idx, 'peak_flux'] = flx
        cutout = apply_transform(cutout, self.transform_function)
        return cutout

    def get_display_data(self, idx):
        """
        Returns a single instance of the dataset in a form that is ready to be
        displayed by the web front end.

        Parameters
        ----------
        idx : str
            Index (should be a string to avoid ambiguity)

        Returns
        -------
        png image object
            Object ready to be passed directly to the frontend
        """

        try:
            img_name = self.metadata.loc[idx, 'original_image']
        except KeyError:
            return None

        this_image = self.images[img_name]
        x0 = self.metadata.loc[idx, 'x']
        y0 = self.metadata.loc[idx, 'y']

        if self.adaptive_sizing:
            sz = self.metadata.loc[idx, 'obj_size']
            if sz < self.min_window_size:
                sz = self.min_window_size
            this_window_size_x = sz
            this_window_size_y = sz
        else:
            this_window_size_x = self.window_size_x
            this_window_size_y = self.window_size_y

        factor = 1.5
        xmin = (int)(x0 - this_window_size_x * factor)
        xmax = (int)(x0 + this_window_size_x * factor)
        ymin = (int)(y0 - this_window_size_y * factor)
        ymax = (int)(y0 + this_window_size_y * factor)

        xstart = max(xmin, 0)
        xend = min(xmax, this_image.metadata['NAXIS1'])

        ystart = max(ymin, 0)
        yend = min(ymax, this_image.metadata['NAXIS2'])
        tot_size_x = int(2 * this_window_size_x * factor)
        tot_size_y = int(2 * this_window_size_y * factor)

        naxis3_present = 'NAXIS3' in this_image.metadata.keys()

        if naxis3_present and this_image.metadata['NAXIS3'] > 1:
            if this_image.metadata['NAXIS3'] != 3:
                shp = [tot_size_y, tot_size_x]
            else:
                shp = [tot_size_y, tot_size_x, this_image.metadata['NAXIS3']]
        else:
            shp = [tot_size_y, tot_size_x]
        cutout = np.zeros(shp)
        # cutout[ystart - ymin:tot_size_y - (ymax - yend),
        #        xstart - xmin:tot_size_x - (xmax - xend)] = img[ystart:yend,
        #
        #                                      xstart:xend]

        img_data = this_image.get_image_data(ystart, yend, xstart, xend)
        cutout[ystart - ymin:yend - ymin,
               xstart - xmin:xend - xmin] = img_data
        cutout = np.nan_to_num(cutout)

        cutout = apply_transform(cutout, self.display_transform_function)

        if len(cutout.shape) > 2 and cutout.shape[-1] >= 3:
            new_cutout = np.zeros([cutout.shape[0], cutout.shape[1], 3])
            new_cutout[:, :, 0] = cutout[:, :, self.bands_rgb['r']]
            new_cutout[:, :, 1] = cutout[:, :, self.bands_rgb['g']]
            new_cutout[:, :, 2] = cutout[:, :, self.bands_rgb['b']]
            cutout = new_cutout

        if self.plot_square:
            offset_x = (tot_size_x - this_window_size_x) // 2
            offset_y = (tot_size_y - this_window_size_y) // 2
            x1 = offset_x
            x2 = tot_size_x - offset_x
            y1 = offset_y
            y2 = tot_size_y - offset_y

            mx = cutout.max()
            cutout[y1:y2, x1] = mx
            cutout[y1:y2, x2] = mx
            cutout[y1, x1:x2] = mx
            cutout[y2, x1:x2] = mx

        min_edge = min(cutout.shape[:2])
        max_edge = max(cutout.shape[:2])

        if self.display_image_size is None:
            self.display_image_size = max(cutout.shape)

        if max_edge != self.display_image_size:
            new_max = self.display_image_size
            new_min = int(min_edge * new_max / max_edge)
            if cutout.shape[0] <= cutout.shape[1]:
                new_shape = [new_min, new_max]
            else:
                new_shape = [new_max, new_min]
            if len(cutout.shape) > 2:
                new_shape.append(cutout.shape[-1])
            cutout = resize(cutout, new_shape, anti_aliasing=True)

        return convert_array_to_image(cutout, plot_cmap=self.plot_cmap,
                                      interpolation=self.display_interpolation)


class ImageThumbnailsDataset(Dataset):
    def __init__(self, display_image_size=None, display_interpolation=None, 
                 transform_function=None,
                 display_transform_function=None, fits_format=False,
                 catalogue=None, check_corrupt_data=False,
                 additional_metadata=None, df_kids=None, type_plot_display=None,
                **kwargs):
        """
        Read in a set of images that have already been cut into thumbnails. 
        This would be uncommon with astronomical data but is needed to read a 
        dataset like galaxy zoo. Inherits from Dataset class.

        Parameters
        ----------
        filename : str
            If a single file (of any time) is to be read from, the path can be
            given using this kwarg. 
        directory : str
            A directory can be given instead of an explicit list of files. The
            child class will load all appropriate files in this directory.
        list_of_files : list
            Instead of the above, a list of files to be loaded can be
            explicitly given.
        output_dir : str
            The directory to save the log file and all outputs to. Defaults to
        display_image_size : None or int
            The size of the image to be displayed on the
            web page. If the image is smaller than this, it will be
            interpolated up to the higher number of pixels. If larger, it will
            be downsampled. If None, will default to the existing width of the
            image.
        display_interpolation : None or str
            Passed to matplotlib imshow. If you are displaying images at a 
            similar size to their native resolution interpolation isn't usually
            necessary. But if they are being upsampled with a larger 
            display_image_size, 'bicubic' will allow for a smoother looking 
            image.
        transform_function : function or list, optional
            The transformation function or list of functions that will be 
            applied to each cutout. The function should take an input 2d array 
            (the cutout) and return an output 2d array. If a list is provided, 
            each function is applied in the order of the list.
        fits_format : boolean
            Set to True if the cutouts are in fits format (as opposed to jpeg
            or png).
        catalogue : pandas.DataFrame or similar
            A catalogue of the positions of sources around which cutouts will
            be extracted. Note that a cutout of size "window_size" will be
            extracted around these positions and must be the same for all
            sources. 
        """

        super().__init__(transform_function=transform_function,
                         display_image_size=display_image_size, 
                         display_interpolation=display_interpolation,
                         catalogue=catalogue,
                         fits_format=fits_format,
                         check_corrupt_data=check_corrupt_data,
                         display_transform_function=display_transform_function,
                         additional_metadata=additional_metadata,
                         df_kids= df_kids, 
                         type_plot_display = type_plot_display,
                         **kwargs)

        self.data_type = 'image'
        self.known_file_types = ['png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff',
                                 'fits', 'fits.fz', 'fits.gz', 'FITS',
                                 'FITS.fz', 'FITS.gz'
                                 ]
        
        self.transform_function = transform_function
        self.check_corrupt_data = check_corrupt_data

        if display_transform_function is None:
            self.display_transform_function = self.transform_function
        else:
            self.display_transform_function = display_transform_function

        if df_kids is not None:
            print('dataframe from KiDS')
            is_kids = True
        else:
            is_kids = False

        self.display_image_size = display_image_size
        self.display_interpolation = display_interpolation
        self.fits_format = fits_format
        self.is_kids = is_kids
        self.type_plot_display =  type_plot_display 
       

        if catalogue is not None and is_kids==False:
            if 'objid' in catalogue.columns:
                catalogue.set_index('objid')
                catalogue.index = catalogue.index.astype(
                    str) + '_' + catalogue.groupby(
                        level=0).cumcount().astype(str)
            self.metadata = catalogue
        elif catalogue is None and is_kids==False:
            inds = []
            file_paths = []
           
            for f in self.files:
                extension = f.split('.')[-1]
                if extension in self.known_file_types:
                    inds.append(
                        f.split(os.path.sep)[-1][:-(len(extension) + 1)])
                    file_paths.append(f)
                self.metadata = pd.DataFrame(index=inds,
                                        data={'filename': file_paths})
        elif is_kids==True:

            #this works for the teglie subset, for bigger dataset i cant load them all together
            if False:
                with open('/home/grespanm/github/data/image_data.pkl', 'rb') as file:
                    self.df_img_kids = pickle.load(file)
            
            #pd.read_csv(os.path.join('/home/grespanm/github/data','table_all_checked_with_img.csv'))
            #print(f"loaded df - { self.df_img_kids.loc[0,'image']}")
            '''
            if 'df_list_obj' in kwargs:
                df_list_obj = kwargs['df_list_obj']
            else: 
                raise ValueError('Dataframe with objects (ID and KIDS_TILE) needed')
            '''
            if 'ID' in df_kids.columns and 'KIDS_TILE' in df_kids.columns:
                df_kids.rename(columns={"ID": "KIDS_ID"}, inplace=True)
            elif 'KIDS_ID' in df_kids.columns and 'KIDS_TILE' in df_kids.columns:
                pass
            else:
                print("DataFrame columns:", df_kids.columns.tolist())
                print(df_kids.head())
                raise ValueError('Dataframe must have columns "ID" (or "KIDS_ID") and "KIDS_TILE".')

            
            #if 'list_of_files' not in df_list_obj.columns:
            #    raise ValueError('Dataframe with images path needed (list_of_files column)')

            #if 'FOLDER' not in df_kids.columns:
            #    raise ValueError('Dataframe with images path needed (FOLDER column)')

            #file_paths = df_kids['FOLDER'].values

            ##label from transformer encoder
            if 'LABEL' in  df_kids.columns:
                pass
            else:
                df_kids['LABEL'] = np.ones(len(df_kids))*-1

            #label from active learning
            if 'LABEL_AL' in  df_kids.columns:
                pass
            else:
                df_kids['LABEL_AL'] = np.ones(len(df_kids))*-1
            
            inds =  df_kids['KIDS_ID'].values
            df_kids['filename'] = [tile +'__'+ id for id,tile in zip(df_kids['KIDS_ID'].values,df_kids['KIDS_TILE'].values)]
    
            #df_kids = df_kids.drop_duplicates(subset='KIDS_ID').reset_index(drop=True)
            #print(df_kids.head())
            self.metadata = df_kids.set_index(inds)
            #df_kids.to_csv(os.path.join('/Users/mrgr/Documents/GitHub/KiDS_astronomaly/example_data/KiDS_cutouts','aa.csv'), index=False)
                
        self.index = self.metadata.index.values

        if additional_metadata is not None:
            self.metadata = self.metadata.join(additional_metadata)

    def get_sample(self, idx):
        """
        Returns the data for a single sample in the dataset as indexed by idx.

        Parameters
        ----------
        idx : string
            Index of sample

        Returns
        -------
        nd.array
            Array of image cutout
        """

        print(f'get sample  {self.fits_format} and {(not self.is_kids)}')

        if self.fits_format and (not self.is_kids):
            try:
                filename = self.metadata.loc[idx, 'filename']
                img = fits.getdata(filename, memmap=True)
                #print(np.shape(img))
                #print(type(img))
                return apply_transform(img, self.transform_function)

            except TypeError:
                msg = "TypeError cannot read image: Corrupt file"
                logging_tools.log(msg, level="ERROR")

                if self.check_corrupt_data:
                    utils.remove_corrupt_file(
                        self.index, self.metadata.index, idx)
                else:
                    print('Corrupted data: Enable check_corrupt_data.')

            except OSError:
                msg = "OSError cannot read image: Empty file"
                logging_tools.log(msg, level="ERROR")

                if self.check_corrupt_data:
                    utils.remove_corrupt_file(
                        self.index, self.metadata.index, idx)
                else:
                    print('Missing data: Enable check_corrupt_data.')

        elif self.fits_format and self.is_kids:
            '''
            # code to load the fits  file for 3 channels
            try:    
                img = from_fits_to_array(self.metadata.loc[idx, 'FOLDER'],
                            self.metadata.loc[idx,'KIDS_ID'] ,self.metadata.loc[idx, 'KIDS_TILE'],  
                            channels=['r', 'i','g'], astronomaly=True)
                if len(np.shape(img))<2:
                    #print('return None')
                    return np.empty((101,101,3))
                #print(np.shape(img))
                img = np.transpose(img, (1,2,0)) 

                #print(np.shape(img))
                #img = image_preprocessing.image_transform_greyscale(img) 
                #print('load',np.shape(apply_transform(img.copy(), self.transform_function)))       
            
            except FileNotFoundError:
                print('File not found')
                pass
            except ValueError:
                print('Value error, the same of the img is', np.shape(img))
            '''
            #print('getting img from dataframe')
            img =  self.df_img_kids[idx]
            # self.metadata.loc[idx, 'image']
            #print(img)
            return apply_transform(img.copy(), self.transform_function)
        else:
            filename = self.metadata.loc[idx, 'filename']
            img = cv2.imread(filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            print(np.shape(img))
            return apply_transform(img, self.transform_function)

    def get_display_data(self, idx):
        """
        Returns a single instance of the dataset in a form that is ready to be
        displayed by the web front end.

        Parameters
        ----------
        idx : str
            Index (should be a string to avoid ambiguity)

        Returns
        -------
        png image object
            Object ready to be passed directly to the frontend
        """
       
        if self.fits_format:
            if not self.is_kids:
                try:
                    filename = self.metadata.loc[idx, 'filename']
                    cutout = fits.getdata(filename, memmap=True)

                except TypeError:
                    msg = "TypeError cannot read image: Corrupted file"
                    logging_tools.log(msg, level="ERROR")

                    if self.check_corrupt_data:
                        sz = self.display_image_size
                        if sz is None:
                            sz = 256  # Just set to a sensible default
                        cutout = np.zeros([1, sz, sz], dtype=int)
                    else:
                        print('Corrupted data: Enable check_corrupt_data.')

                except OSError:
                    msg = "OSError cannot read image: Empty file"
                    logging_tools.log(msg, level="ERROR")

                    if self.check_corrupt_data:
                        sz = self.display_image_size
                        if sz is None:
                            sz = 256  # Just set to a sensible default
                        cutout = np.zeros([1, sz, sz], dtype=int)
                    else:
                        print('Missing data: Enable check_corrupt_data.')
            else:
                    #print('FOLDERRR',self.metadata.loc[idx, 'FOLDER'])
                    #print('FOLDERRR', self.metadata.loc[idx, 'KIDS_ID'] ,self.metadata.loc[idx, 'KIDS_TILE'])
                    #is img from kids, load it the correct  way 
                    '''
                    cutout =  self.df_img_kids.loc[idx, 'image'] 
                    print(idx)
                    print(type(cutout))
                    print(cutout.shape)
                    
                    img = from_fits_to_array(settings.path_to_save_imgs,
                        #self.metadata.loc[idx, 'FOLDER'],
                                self.metadata.loc[idx, 'KIDS_ID'] ,self.metadata.loc[idx, 'KIDS_TILE'],  
                                channels=['r', 'i','g'],  astronomaly=True)
                    #print(np.shape(img))
                    #cutout = np.transpose(img.copy(), (1,2,0))
                    '''
                    hdf5_file_path_1 = '/home/grespanm/github/data/dr4_cutouts.h5' if os.path.exists('/home/grespanm/github/data/dr4_cutouts.h5') else '/home/astrodust/mnt/github/data/dr4_cutouts.h5'
                    hdf5_file_path_2 = '/home/grespanm/github/data/dr4_cutouts_2.h5' if os.path.exists('/home/grespanm/github/data/dr4_cutouts_2.h5') else '/home/astrodust/mnt/github/data/dr4_cutouts_2.h5'

                    kids_ids_file_1 = '/home/grespanm/github/data/dr4_cutoutsKIDS_ID_1.txt' if os.path.exists('/home/grespanm/github/data/dr4_cutoutsKIDS_ID_1.txt') else '/home/astrodust/mnt/github/data/dr4_cutoutsKIDS_ID_1.txt'
                    kids_ids_file_2 = '/home/grespanm/github/data/dr4_cutoutsKIDS_ID_2.txt' if os.path.exists('/home/grespanm/github/data/dr4_cutoutsKIDS_ID_2.txt') else '/home/astrodust/mnt/github/data/dr4_cutoutsKIDS_ID_2.txt'


                    # Load IDs from the text files into sets
                    with open(kids_ids_file_1, 'r') as f:
                        ids_set_1 = set(line.strip() for line in f)

                    with open(kids_ids_file_2, 'r') as f:
                        ids_set_2 = set(line.strip() for line in f)


                    # Check the first file
                    cutout = None
                    if idx in ids_set_1:
                        with h5py.File(hdf5_file_path_1, 'r') as hdf5_file:
                            if idx in hdf5_file:
                                cutout = hdf5_file[idx][:]

                    # Check the second file if not found
                    if cutout is None and idx in ids_set_2:
                        with h5py.File(hdf5_file_path_2, 'r') as hdf5_file:
                            if idx in hdf5_file:
                                cutout = hdf5_file[idx][:]    

                    #this works with the pickle file 
                    #cutout =  self.df_img_kids[idx]

                    #with open('/home/grespanm/github/data/image_data.pkl', 'rb') as file:
                    #    df_img_kids = pickle.load(file)
                    #df_img_kids

            
                    
            
        else:
            print('else')
            filename = self.metadata.loc[idx, 'filename']
            cutout = cv2.imread(filename)
            cutout = cv2.cvtColor(cutout, cv2.COLOR_BGR2RGB)
            # Because we're usually working with data in numpy arrays and 
            # not normally png images, we need to flip them to align with
            # how Astronomaly displays them in the front end
            cutout = np.flip(cutout, axis=0)

        cutout = apply_transform(cutout, self.display_transform_function)

       
        if self.display_image_size is None:
            self.display_image_size = max((cutout.shape[0], cutout.shape[1]))

        min_edge = min(cutout.shape[:2])
        max_edge = max(cutout.shape[:2])


        if max_edge != self.display_image_size:
            new_max = self.display_image_size
            new_min = int(min_edge * new_max / max_edge)
            if cutout.shape[0] <= cutout.shape[1]:
                new_shape = [new_min, new_max]
            else:
                new_shape = [new_max, new_min]
            if len(cutout.shape) > 2:  
                new_shape.append(cutout.shape[-1])
            cutout = resize(cutout, new_shape, anti_aliasing=True)

        return convert_array_to_image(cutout, 
                                      interpolation=self.display_interpolation)

    def fits_to_png(self, scores):
        """
        Simple function that outputs png files from the input fits files

        Parameters
        ----------
        Scores : string
            Score of sample

        Returns
        -------
        png : image object
            Images are created and saved in the output folder
        """

        for i in range(len(scores)):
            idx = scores.index[i]

            filename = self.metadata.loc[idx, 'filenames']
            flux = self.metadata.loc[idx, 'peak_flux']
            for root, directories, f_names in os.walk(self.directory):
                if filename in f_names:
                    file_path = os.path.join(root, filename)

            output_path = os.path.join(self.output_dir, 'PNG', 'Anomaly Score')

            if not os.path.exists(output_path):
                os.makedirs(output_path)

            data = fits.getdata(file_path, memmap=True)

            if len(np.shape(data)) > 2:
                one = data[0, :, :]
                two = data[1, :, :]
                three = data[2, :, :]
                data = np.dstack((three, two, one))
                transformed_image = apply_transform(
                    data, self.display_transform_function)
            else:
                transformed_image = apply_transform(
                    data, self.display_transform_function)

            plt.imsave(output_path + '/AS:' + '%.6s' % scores.score[i] + 
                       '_NAME:' + str(idx) + 
                       '_FLUX:' + '%.4s' % flux + '.png', transformed_image)
