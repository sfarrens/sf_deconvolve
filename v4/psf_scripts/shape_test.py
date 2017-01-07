#!/Users/sfarrens/Documents/Library/anaconda/bin/python

import numpy as np
import argparse as ap
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from functions.astro import rad2deg
from creepy.image.shape import Ellipticity


def get_opts():

    '''Get script arguments'''

    global opts

    parser = ap.ArgumentParser('SCRIPT OPTIONS:',
                               formatter_class=ap.RawTextHelpFormatter)

    parser.add_argument('-r', '--rec_data', dest='rec_data',
                        required=True, help='Reconstruction data file name.')

    parser.add_argument('-c', '--clean_data', dest='clean_data',
                        required=True, help='Clean data file name.')

    parser.add_argument('-b', '--rec_back', dest='rec_back',
                        required=False, action='store_true',
                        help='Use reconstruction background.')

    parser.add_argument('--radius', dest='radius', default=15.0, type=float,
                        required=False, help='Line radius.')

    opts = parser.parse_args()


def get_theta(ellip):

    '''Calculate theta from ellipticities'''

    return 0.5 * np.arctan(ellip[1] / ellip[0])


def get_xy(radius, theta):

    '''Calculate x and y from theta and radius'''

    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    x = [-x, x]
    y = [-y, y]

    return x, y


def make_plot():

    # Read data files.
    data_clean = np.load(opts.clean_data)
    data_rec = np.load(opts.rec_data)

    # Calculate ellipticities.
    ellip_clean = np.array([Ellipticity(x).e for x in data_clean]).T
    ellip_rec = np.array([Ellipticity(x).e for x in data_rec]).T

    # Calculate angles.
    theta_clean = get_theta(ellip_clean)
    theta_rec = get_theta(ellip_rec)

    # Calculate the difference between the clean and rec angles.
    diff = [np.linalg.norm(a - b) for a, b in zip(theta_clean, theta_rec)]

    # Set plot options.
    fig = plt.figure(figsize=(8, 11))
    cmap = 'Spectral_r'
    extent = (-20, 20, -20, 20)
    grid = gridspec.GridSpec(5, 5, wspace=0.2, hspace=0.2)

    # Make the plot.
    for i in range(25):

        x1, y1 = get_xy(opts.radius, theta_clean[i])
        x2, y2 = get_xy(opts.radius, theta_rec[i])

        ax = plt.subplot(grid[i])
        if opts.rec_back:
            ax.imshow(data_rec[i], cmap=cmap, extent=extent)
        else:
            ax.imshow(data_clean[i], cmap=cmap, extent=extent)
        ax.plot(x1, y1, '--', color='0.75')
        ax.plot(x2, y2, ':', color='0.5')
        ax.set_title('Diff = ' + str(np.round(diff[i], 2)))
        ax.axis('off')

    # Save file to output.
    file_name = opts.rec_data + '_check_ellip.pdf'
    plt.savefig(file_name)
    print 'Output saved to:', file_name


def main():

    get_opts()
    make_plot()


if __name__ == "__main__":
    main()
