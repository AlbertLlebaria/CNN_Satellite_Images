from scipy import ndimage, misc, signal
import numpy as np
import rasterio
from rasterio.plot import show
import os
from modules import utils
from modules.geofolki.algorithm import GEFolki
from modules.geofolki.tools import wrapData
import matplotlib.pyplot as plt


rank = 4
rad = 32


# def rank_filter_sup(I, rad):
#     nl, nc = I.shape
#     R = np.zeros([nl, nc])
#     for i in range(-rad, rad+1):  # indice de ligne
#         for j in range(-rad, rad+1):  # indice de colonne
#             if i != 0:
#                 if i < 0:
#                     tmp = np.concatenate(
#                         [I[-i:, :], np.zeros([-i, nc])], axis=0)
#                 else:
#                     tmp = np.concatenate(
#                         [np.zeros([i, nc]), I[:-i, :]], axis=0)
#             else:
#                 tmp = I
#             if j != 0:
#                 if j < 0:
#                     tmp = np.concatenate(
#                         [tmp[:, -j:], np.zeros([nl, -j])], axis=1)
#                 else:
#                     tmp = np.concatenate(
#                         [np.zeros([nl, j]), tmp[:, :-j]], axis=1)

#             idx = (tmp > I)
#             R[idx] = R[idx]+1

#     return R


# def rank_filter_inf(I, rad):
#     nl, nc = I.shape
#     R = np.zeros([nl, nc])
#     for i in range(-rad, rad+1):
#         for j in range(-rad, rad+1):
#             if i != 0:
#                 if i < 0:  # on decalle vers le haut de i lignes
#                     tmp = np.concatenate(
#                         [I[-i:, :], np.zeros([-i, nc])], axis=0)
#                 else:
#                     tmp = np.concatenate(
#                         [np.zeros([i, nc]), I[:-i, :]], axis=0)
#             else:
#                 tmp = I
#             if j != 0:
#                 if j < 0:
#                     tmp = np.concatenate(
#                         [tmp[:, -j:], np.zeros([nl, -j])], axis=1)
#                 else:
#                     tmp = np.concatenate(
#                         [np.zeros([nl, j]), tmp[:, :-j]], axis=1)

#             idx = (tmp < I)
#             R[idx] = R[idx]+1
#     return R


# def interp2(I, x, y): return ndimage.map_coordinates(
#     I, [y, x], order=1, mode='nearest').reshape(I.shape)


# def conv2bis(I, w): return signal.convolve2d(I, w, mode='valid')


# def conv2SepMatlabbis(I, fen):

#     rad = int((fen.size-1)/2)
#     ligne = np.zeros((rad, I.shape[1]))
#     I = np.append(ligne, I, axis=0)
#     I = np.append(I, ligne, axis=0)

#     colonne = np.zeros((I.shape[0], rad))
#     I = np.append(colonne, I, axis=1)
#     I = np.append(I, colonne, axis=1)

#     res = conv2bis(conv2bis(I, fen.T), fen)
#     return res


# def wrapData(I, u, v):
#     '''
#         Apply the [u,v] optical flow to the data I
#     '''
#     col, row = I.shape[1], I.shape[0]
#     X, Y = np.meshgrid(range(col), range(row))
#     R = interp2(I, X+u, Y+v)
#     return R


# def filter_image(img, f=0, rank=4, radius=16, smoothing=4):
#     """
#     Function that applies the image transformations f0 and f1. These function consist of a sequence of tranformations:
#         f0 = R ◦ g
#         f1 =C◦R◦g

#     Where C = Local Contrast Inversion
#     Where g =  Rolling guidance filter
#     Where R = Rank filter
#     """
#     if f is 1:
#         C = equalize_adapthist(img, 8, clip_limit=1, nbins=256)
#         R = ndimage.rank_filter(C, rank, size=radius)
#         G = ndimage.gaussian_filter(R, 6)
#         H0 = guided_filter(img, G, radius, smoothing)
#     else:
#         R = ndimage.rank_filter(img, rank, size=radius)
#         H0 = ndimage.gaussian_filter(R, 6)
#     return H0


# optical = rasterio.open('data/train_val/tile_4_rgb.tif')
# optical_data = optical.read([1])
# optical_data = optical_data[0].astype(np.uint8)
# data_profile = optical.profile.copy()
# data_profile.update({'count': 1})

# DEM = rasterio.open('data/train_val/tile_4_mask.tif')
# DEM_data = DEM.read([1])
# DEM_data = DEM_data[0].astype(np.uint8)


# R0 = rank_filter_sup(DEM_data, rank)
# R1i = rank_filter_inf(optical_data, rank)
# R1s = rank_filter_sup(optical_data, rank)


# I0 = filter_image(optical_data, f=1)
# I1 = filter_image(DEM_data, f=0)


# u = np.zeros(I0.shape)
# v = np.zeros(I1.shape)
# Iy, Ix = np.gradient(R0)


# show(I1)
# show(I0)

# cols, rows = I0.shape[1], I0.shape[0]
# x, y = np.meshgrid(range(cols), range(rows))

# H1 = I1
# H0 = I0


# H1 = H1.astype(np.float32)
# H0 = H0.astype(np.float32)

# burt1D = np.array(np.ones([1, 2*rad+1]))/(2*rad + 1)
# print(burt1D.shape)


# def W(xin): return conv2SepMatlabbis(xin, burt1D)


# Ixx = W(Ix*Ix)
# Iyy = W(Iy*Iy)
# Ixy = W(Ix*Iy)
# D = Ixx*Iyy - Ixy**2


# for i in range(4):
#     dx = x + u
#     dy = y + v
#     dx[dx < 0] = 0
#     dy[dy < 0] = 0
#     dx[dx > cols-1] = cols-1
#     dy[dy > rows-1] = rows-1
#     H1w = interp2(H1, dx, dy)

#     crit1 = conv2SepMatlabbis(np.abs(H0-H1w), np.ones([2*rank+1, 1]))
#     crit2 = conv2SepMatlabbis(np.abs(1-H0-H1w), np.ones([2*rank+1, 1]))
#     R1w = interp2(R1s, x+u, y+v)
#     R1w_1 = interp2(R1i, x+u, y+v)

#     R1w[crit1 > crit2] = R1w_1[crit1 > crit2]
#     it = R0 - R1w + u*Ix + v*Iy
#     Ixt = W(Ix * it)
#     Iyt = W(Iy * it)
#     u = (Iyy * Ixt - Ixy * Iyt)/D
#     v = (Ixx * Iyt - Ixy * Ixt)/D
#     unvalid = np.isnan(u) | np.isinf(u) | np.isnan(v) | np.isinf(v)
#     u[unvalid] = 0
#     v[unvalid] = 0

# N = np.sqrt(u**2+v**2)
# show(N)

# Ioptique_resampled = wrapData(optical_data, u, v).astype(np.uint8)
# show(Ioptique_resampled)

# print("Fin recalage optique/Radar \n\n")
# with rasterio.open('result.tif', 'w', **data_profile) as dst:
#     dst.write(np.array([Ioptique_resampled]))



optical = rasterio.open('data/train_val/train/tile_136_rgb.tif')


optical_data = optical.read()

Ioptical_data = optical_data[0, :, :]
Ioptical_data = Ioptical_data.astype(np.uint8)
data_profile = optical.profile.copy()

DEM = rasterio.open('data/train_val/train/tile_136_mask.tif')
DEM_data = DEM.read([1])
min_dem = np.min(DEM_data.flatten())
max_dem = np.max(DEM_data.flatten())
DEM_data = np.array(
                    list(map(lambda x: ((x-min_dem)/(max_dem-min_dem))*255, DEM_data)))

DEM_data = DEM_data[0].astype(np.uint8)

u, v = GEFolki(DEM_data, Ioptical_data, iteration=6,
               radius=range(4, 16, -4), rank=6, levels=6)

N = np.sqrt(u**2+v**2)

res = np.zeros_like(optical_data)

for i in range(3):
    res[i] = wrapData(optical_data[i], u, v)


print("Fin recalage optique/Radar \n\n")
with rasterio.open('building_6_16_6_6.tif', 'w', **data_profile) as dst:
    dst.write(res)
