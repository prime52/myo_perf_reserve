#!/usr/bin/env python
# coding: utf-8
"""
An automatic estimation of MPRI in DCE cardiac MR images

2022-04-13

YC Kim
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time, os
import scipy
import glob
import pydicom as dicom
import cmath
import gc
import pickle

from skimage.filters import median
from skimage.morphology import disk

from tensorlayer.layers import clear_layers_name
from tensorflow import reset_default_graph

# from binary import dc, hd
from copy import deepcopy
from unet_v2 import u_net
from util.functions import setdiff_mask, get_index_fromMask
from skimage import transform, morphology
from scipy.ndimage.measurements import label
from scipy.optimize import curve_fit
from skimage.measure import regionprops
from sklearn.cluster import KMeans

import joblib
import skimage.morphology as sm
from landmark_detection_v3 import detect_landmark
from upslope_myo_segment_v1 import upslope_myo_segmental


def get_largest_mask(mask1):

	labeled_arr, nfeature = label(mask1)

	props = regionprops(labeled_arr)

	max_area = -1000
	max_ind = 0
	for jj in range(nfeature):
		area = props[jj].filled_area
		# print('region %d, area=%5d'% (jj, area))
		if area > max_area:
			max_area = area
			max_ind = jj

	mask2 = np.zeros(mask1.shape)
	mask2[labeled_arr == max_ind + 1] = 1

	return mask2


def segment_fat(imgtmp):

	img2 = imgtmp
	# maxmap = np.amax(img2, axis=2)
	minmap = np.amin(img2, axis=2)
	nx, ny, _ = img2.shape
	thresh_fat = 0.5 * np.amax(minmap)
	ind_subcutfat = np.where(minmap > thresh_fat)
	mask_subcutfat = np.zeros([nx, ny])
	mask_subcutfat[ind_subcutfat] = 1.0
	disk_diameter = 8   #15
	mask_subcutfat = sm.dilation(mask_subcutfat, sm.disk(disk_diameter))
	ind_fat_ = np.where(mask_subcutfat == 1)  # (1, 2, 3, ..., 4), (3, 4, 5, ..., 6)

	return ind_fat_


def fit_RVtic(s1, ind_RVpeak, type):

	def gauss(x, *p):
		A, mu, sigma, y0 = p
		return A * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2)) + y0

	def gauss2(x, *p):
		A1, mu1, sigma1, y0, A2, mu2, sigma2 = p

		result = A1 * np.exp(-(x - mu1) ** 2 / (2. * sigma1 ** 2)) + A2 * np.exp(
			-(x - mu2) ** 2 / (2. * sigma2 ** 2)) + y0

		return result

	two_peaks = True

	if two_peaks:
		p0 = [np.amax(s1), ind_RVpeak, 5., s1[0], 0.5 * np.amax(s1), ind_RVpeak + 30, 5.]

		popt, var_matrix = curve_fit(f=gauss2, xdata=np.array(range(s1.shape[0])), ydata=s1, p0=p0,
		                             bounds=(
		                             [0.5 * np.amax(s1), ind_RVpeak - 3, 1., s1[0] - 2.0 * s1[0], 0.2 * np.amax(s1),
		                              ind_RVpeak + 30 - 6, 1.],
		                             [1.5 * np.amax(s1), ind_RVpeak + 3, 7., s1[0] + 2.0 * s1[0], 0.7 * np.amax(s1),
		                              ind_RVpeak + 30 + 6, 7.]))

		s1_fit = gauss2(np.array(range(80)), *popt)

	else:
		p0 = [np.amax(s1), ind_RVpeak, 5., s1[0]]

		try:
			popt, var_matrix = curve_fit(f=gauss, xdata=np.array(range(s1.shape[0])), ydata=s1, p0=p0,
			                             bounds=([0.5 * np.amax(s1), ind_RVpeak - 3, 1., s1[0] - 2.0 * s1[0]],
			                                     [1.5 * np.amax(s1), ind_RVpeak + 3, 5., s1[0] + 2.0 * s1[0]]),
			                             method='dogbox')
		except RuntimeError:
			print('Error - curve_fit failed.')

		s1_fit = gauss(np.array(range(80)), *popt)

	return s1_fit


''' Automatically find the peak RV enhanced frame and segment RV cavity using K-means clustering '''

def autoRVseed_cluster(imgdyn, slno, id, type, disp_figure_flag):

	# Apply k-means clustering to TIC curves
	nx, ny, nt, nslice = imgdyn.shape
	imgt_orig = imgdyn[:, :, :, slno]
	imgt_orig2 = deepcopy(imgt_orig)
	ind_fat = segment_fat(imgt_orig)
	# RVseed = (row, col)
	a = 0; b = 0
	imgt = np.zeros(imgt_orig.shape)
	for jj in range(nt):
		tmp = imgt_orig[:, :, jj]
		tmp[ind_fat] = 0
		# trim the edge
		tmp[:, -20:] = 0; tmp[:, :20] = 0; tmp[:20, :] = 0; tmp[-20:, :] = 0
		imgt[:, :, jj] = tmp

	nrow, ncol, nframe = imgt.shape
	imgtn = imgt / imgt.max()  # normalize
	imgt1d = imgtn.transpose(2, 0, 1).reshape(nframe, -1)
	nvoxel = imgt1d.shape[1]
	tic_auc = np.sum(imgt1d, 0)
	ind_big_auc = np.argsort(tic_auc)  # 올림차순
	ind_big_auc = ind_big_auc[-int(0.4 * nvoxel):]
	# tic = imgt1d[:, ind_big_auc[-1]]

	# measure roughness
	roughness = np.zeros(nvoxel)
	for jj in range(nvoxel):
		tic = imgt1d[:, jj]
		ddtic = np.diff(np.diff(tic))
		roughness[jj] = np.sum(np.multiply(ddtic, ddtic))
	ind_big_roughness = np.argsort(roughness)  # 올림차순
	ind_big_roughness = ind_big_roughness[-int(0.1 * nvoxel):]
	ind_candidate = np.intersect1d(ind_big_auc, ind_big_roughness)
	# sub_candidate = np.unravel_index(ind_candidate, (nrow, ncol))

	nvoxel_cand = ind_candidate.shape[0]

	nframe_for_kmeans = 60   # default = nframe

	# data = np.zeros((nframe, nvoxel_cand))
	data = np.zeros((nframe_for_kmeans, nvoxel_cand))

	for ii, jj in enumerate(ind_candidate):
		data[:, ii] = imgt1d[:nframe_for_kmeans, jj]
	data = np.transpose(data, [1, 0])

	ncluster = 4  # cluster 개수, 중요한 파라미터임.

	kmeans = KMeans(n_clusters=ncluster, random_state=0).fit(data)
	labels = kmeans.labels_

	nvoxel_cluster = np.zeros(ncluster)
	# tic_cluster_avg = np.zeros((nframe, ncluster))
	tic_cluster_avg = np.zeros((nframe_for_kmeans, ncluster))
	for kk in range(ncluster):
		ind0 = np.where(labels == kk)
		nvoxel_cluster[kk] = ind0[0].shape[0]
		ind_label0 = ind_candidate[ind0]
		# sub_label0 = np.unravel_index(ind_label0, (nrow, ncol))
		tmp = np.mean(imgt1d[:nframe_for_kmeans, ind_label0], axis=1)
		tic_cluster_avg[:, kk] = tmp / tmp.max()

	# Compare the first moment
	xrange = np.array(range(0, nframe_for_kmeans, 1))
	first_moment_cluster = np.zeros(ncluster)
	peak_index_cluster = np.zeros(ncluster)

	min_max_diff_cluster = np.zeros(ncluster)
	obj_fcn_cluster = np.zeros(ncluster)

	# lambda_ = 1; alpha_ = 100;
	alpha_ = 20; beta_ = 80

	for kk in range(ncluster):
		first_moment_cluster[kk] = np.mean(np.multiply(xrange, np.transpose(tic_cluster_avg[:, kk])))
		peak_index_cluster[kk] = np.argmax(tic_cluster_avg[:,kk])
		min_max_diff_cluster[kk] = np.max(tic_cluster_avg[:, kk]) - np.min(tic_cluster_avg[:, kk])

		''' objective function -- RV cluster를 옳게 찾는데 주효함 '''
		# obj_fcn_cluster[kk] = first_moment_cluster[kk] + lambda_ * peak_index_cluster[kk] + alpha_ * 1/nvoxel_cluster[kk] + beta_ * 1/min_max_diff_cluster[kk]
		obj_fcn_cluster[kk] = first_moment_cluster[kk] + alpha_ * peak_index_cluster[kk] + beta_ * 1 / min_max_diff_cluster[kk]

	kk_RV = np.argsort(obj_fcn_cluster)[0]  # 오름차순으로 정렬 후 맨 처음 엘레먼트를(즉, 최소치) 취함
	print(' number of voxels in each cluster: ', nvoxel_cluster)
	print(' cost in each cluster:             ', obj_fcn_cluster, kk_RV)
	ind0 = np.where(labels == kk_RV)
	ind_label0 = ind_candidate[ind0]
	sub_RV = np.unravel_index(ind_label0, (nrow, ncol))
	# print('sub_RV[0], ', sub_RV[0])
	mask_RV = np.zeros((nrow, ncol))
	mask_RV[sub_RV] = 1

	# cluster island 중 가장 큰 부분만 취하기.
	mask_RV = get_largest_mask(mask_RV)
	mask_RV = median(mask_RV, selem=disk(5))

	J_RV_ = mask_RV.reshape((nrow*ncol, 1))
	ind_RV = np.where(J_RV_ == 1)[0]
	ind3 = np.where(mask_RV == 1)
	sigRV = np.zeros(nt)
	for tt in range(nt):
		tmp = imgt_orig[:, :, tt]
		sigRV[tt] = np.mean(tmp[ind3])
	# print('kk')

	filter_sigRV = False
	if filter_sigRV:
		sigRV_filtered = scipy.ndimage.median_filter(sigRV, size=5)
	else:
		sigRV_filtered = sigRV
	ind_RVpeak = np.argmax(sigRV_filtered)
	print(f'ind_RVpeak = {ind_RVpeak}')

	RV_seedy = np.mean(sub_RV[0])
	RV_seedx = np.mean(sub_RV[1])
	RVseed = (RV_seedx, RV_seedy)
	# print('ll')
	imgRV0 = imgt_orig2[:, :, ind_RVpeak]

	print(imgRV0.shape)
	imgRV0 = np.uint32(imgRV0)

	# region growing 수행한 후 mask_RV를 RV cavity로 확장하기!!
	print(' ind3[0].shape ', ind3[0].shape)

	n2 = ind3[0].shape[0]
	seed = (int(ind3[0][int(n2/2)]), int(ind3[1][int(n2/2)])) # (int(sub_RV), int(sub_RV))
	print(' seed = ', seed)
	# print(' image intensity = ', imgRV0[seed])
	# print(imgRV0.dtype)

	# plt.figure(); plt.imshow(imgRV0, cmap='gray'); plt.show()

	imgRV_255 = imgRV0/imgRV0.max() * 255.0
	imgRV_255 = np.uint8(imgRV_255)
	print(' image intensity = ', imgRV_255[seed])

	if disp_figure_flag:
		plt.figure()
		plt.subplot(221); plt.imshow(imgRV_255, cmap='gray', clim=(0, 0.7 * imgRV0.max())); plt.scatter(RVseed[0], RVseed[1], marker='+', c='r');	plt.title('ID=%s, %s' % (id, type))
		plt.subplot(222); plt.imshow(mask_RV, cmap='gray' ); plt.scatter(RVseed[0], RVseed[1], marker='+', c='r');	plt.title('RV identification by clustering')
		plt.subplot(223); plt.plot(range(nt), sigRV, c='b'); plt.plot(range(nt), sigRV_filtered, c='c'); plt.plot(ind_RVpeak, sigRV_filtered[ind_RVpeak], 'ro'); plt.axis([0, nt, 0, 1.05*sigRV_filtered.max()])
		plt.subplot(224); color_set = ['b', 'r', 'k', 'c', 'm']
		for kk in range(ncluster):
			plt.plot(range(nframe_for_kmeans), tic_cluster_avg[:, kk], c=color_set[kk], label='index=' + str(kk))
		plt.legend(['0', '1', '2', '3'])

		plt.show()

	print(f'  np.amax(sigRV_filtered) = {np.amax(sigRV_filtered)}')

	gaussian_fit = False
	if gaussian_fit:
		sigRV_fit = fit_RVtic(sigRV_filtered, ind_RVpeak, type)
	else:
		sigRV_fit = sigRV_filtered

	ind_RVpeak2 = np.argmax(sigRV_fit)

	if False:

		imgRV = imgt_orig2[:, :, ind_RVpeak2]

		plt.figure()
		plt.subplot(221)
		plt.imshow(imgRV, cmap='gray', clim=(0, 0.7*imgRV.max()))
		plt.scatter(RVseed[0], RVseed[1], marker='+', c='r'); plt.title('id=%s, %s' % (id, type))

		plt.subplot(222)
		plt.imshow(mask_RV, cmap='gray')
		plt.scatter(RVseed[0], RVseed[1], marker='+', c='r'); plt.title('RV identification by clustering')

		plt.subplot(223)
		plt.plot(range(nt), sigRV, c='b'); plt.plot(ind_RVpeak, sigRV[ind_RVpeak], 'ro')
		plt.plot(range(nt), sigRV_fit, c='r'); plt.plot(ind_RVpeak2, sigRV_fit[ind_RVpeak2], 'go')

		plt.subplot(224)
		color_set = ['b', 'r', 'k', 'c', 'm']
		for kk in range(ncluster):
			plt.plot(range(nt), tic_cluster_avg[:, kk], c=color_set[kk], label='index='+str(kk))
		plt.title('TIC from an RV ROI')

		# np.savez('test_rvcluster2.npz', imgt=imgt)
		plt.show()

	return RVseed, ind_RVpeak2, a, b, ind_RV, ind_fat  # ind_fat: N x 1 array




def read_dcm(dir1, stress_flag):

	if stress_flag:
		dir2 = os.path.join(dir1, 'perf/stress')
	else:  # rest
		dir2 = os.path.join(dir1, 'perf/rest')

	if os.path.isdir(dir2):
		list_dirs = os.listdir(dir2)
		perf = {}
		perf['series'] = []
		perf['nslice'] = 4
		nframe = 80
		
		for count, dn in enumerate(list_dirs):
			perf['series'].append(dn)
			
		perf['nslice'] = count + 1

		for slno in range(perf['nslice']):

			b = os.path.join(dir2, perf['series'][slno])

			dicom_list = os.listdir(b)

			if dicom_list[4].startswith('IMG0'):
				dcm_files = glob.glob(b+'/IMG0*')
			elif dicom_list[4].endswith('.dcm'):
				dcm_files = glob.glob(b+'/*.dcm')
			else:
				print(f'slno={slno}: dicom file name is not valid. quit!')
				quit()

			for frameno, f in enumerate(dcm_files):
				# print('\t\t slno = %d, frameno = %d, fname=%s ' % (slno, frameno, f))
				ds = dicom.read_file(os.path.join(b, f), force=True)

				if slno == 0 and frameno == 0:
					# print(ds)
					print('Rows = %d, Columns = %d' % (ds.Rows, ds.Columns))

					if ds.Rows < 200:
						nrow = int(2*ds.Rows)
						ncol = int(2*ds.Columns)
						pixelDims = (nrow, ncol, int(nframe), int(perf['nslice']))
						perf['PixelSpacing'] = ds.PixelSpacing
						perf['PixelSpacing'][0] = 0.5 * ds.PixelSpacing[0]
						perf['PixelSpacing'][1] = 0.5 * ds.PixelSpacing[1]
					else:
						nrow = int(ds.Rows)
						ncol = int(ds.Columns)
						pixelDims = (nrow, ncol, nframe, int(perf['nslice']))
						perf['PixelSpacing'] = ds.PixelSpacing

					perf['images'] = np.zeros(pixelDims, dtype=ds.pixel_array.dtype)
					perf['img_myoenhance'] = np.zeros((nrow, ncol, int(perf['nslice'])))
					perf['mask_myocardium'] = np.zeros((nrow, ncol, int(perf['nslice'])))
					perf['mask_defect'] = np.zeros((nrow, ncol, int(perf['nslice'])))
					perf['mask_myoseg_index'] = np.zeros((nrow, ncol, int(perf['nslice'])))
					perf['patientID'] = ds.PatientID
					perf['flipangle'] = ds.FlipAngle
					perf['invtime'] = ds.InversionTime
					perf['npe'] = ds.NumberOfPhaseEncodingSteps

					perf['zLVc'] = np.zeros(perf['nslice'], dtype=np.complex_)
					perf['zRVi'] = np.zeros(perf['nslice'], dtype=np.complex_)

					perf['SliceLocation'] = np.zeros(perf['nslice'])

				if ds.Rows < 200:
					img = ds.pixel_array
					perf['images'][:, :, frameno, slno] = scipy.misc.imresize(img, int(200), interp='bicubic')
				else:
					if slno > 0 and ds.Rows != pixelDims[0]:
						# print('rotate image by 90 deg.')
						perf['images'][:, :, frameno, slno] = np.flipud(np.transpose(ds.pixel_array))
					else:
						perf['images'][:, :, frameno, slno] = ds.pixel_array

				# if slno == 0:
				# 	perf['RRinterval'][frameno] = ds.NominalInterval * (10**-3)

				if frameno == 0:
					perf['SliceLocation'][slno] = ds.SliceLocation
					# perf['TriggerTime'][slno] = ds.TriggerTime
					perf['SeriesDescription'] = ds.SeriesDescription
					print(f'SeriesDescription : {ds.SeriesDescription}')


		img0 = perf['images']
		print('\t slice location = ', perf['SliceLocation'])
		sliceloc_arr = np.absolute(perf['SliceLocation'])

		if np.diff(sliceloc_arr)[0] < 0:
			print('  slice order: from apical to basal. reverse the slice order and save the images.')
			img0 = img0[:, :, :, ::-1]
			perf['images'] = img0

	return perf


def segment_myo(img0, slno, device_name, id, type):

	""" 사용자 초이스 """
	unet_fname = './unet_model/UNet_n=110_scale01_epoch=30'
	# thresh = 0.3
	nframe_metric = 30  # metric 계산에 이용되는 프레임 수. default = 20
	niter_mc = 100  # default = 100

	nrow_des, ncol_des = 256, 216  # desirable dimensions for U-net
	# nrow_orig = img0.shape[0];	ncol_orig = img0.shape[1]
	nrow_orig, ncol_orig = img0.shape[0], img0.shape[1]
	# img0_nomask = deepcopy(img0)

	'''  crop image and mask to 256 x 216  '''
	img = img0[int(nrow_orig/2 - nrow_des/2):int(nrow_orig/2 + nrow_des/2), int(ncol_orig/2 - ncol_des/2):int(ncol_orig/2 + ncol_des/2), :, :]
	# imgdyn = img[:, :, :, slno]
	# img_nomask = img0_nomask[int(nrow_orig/2 - nrow_des/2):int(nrow_orig/2 + nrow_des/2), int(ncol_orig/2 - ncol_des/2):int(ncol_orig/2 + ncol_des/2), :, :]
	img2 = deepcopy(img)
	nframe = img2.shape[2]
	RVseed, fno_RV, _, _, indRVcavity, ind_fat = autoRVseed_cluster(img0, slno, id, type, disp_figure_flag)
	fno_LV = fno_RV + 4

	with tf.device(device_name):
		t_image = tf.placeholder('float32', [1, nrow_des, ncol_des, 1], name='input_image')
		net = u_net(t_image, is_train=False, apply_dropout=True, reuse=False, n_out=2)

	saver = tf.train.Saver()
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
	config = tf.ConfigProto(gpu_options=gpu_options)
	config.gpu_options.allow_growth = True
	sess = tf.InteractiveSession(config=config)
	saver.restore(sess, unet_fname)

	# n_std_high1 = []; 	n_std_high2 = []
	sum_std_fn_endo = []; 	sum_std_fn_epic = []
	# sum_mean_fn_endo = [];  sum_mean_fn_epic = []

	shape = [img.shape[2], img.shape[0], img.shape[1]]
	input_img = np.zeros(shape); 	mean_endo = np.zeros(shape); 	mean_epic = np.zeros(shape)
	sdev_endo = np.zeros(shape); 	sdev_epic = np.zeros(shape)

	# Monte Carlo prediction
	def monte_carlo_predict(sess, image, iters):
		evaluations = []
		for _ in range(iters):
			out = sess.run(net.outputs, {t_image: image[np.newaxis, :, :, np.newaxis]})
			# out의 shape = (1, 256, 216, 2)
			evaluations.append(out)
		mean = np.mean(evaluations, axis=0)
		stddev = np.std(evaluations, axis=0)

		return mean[0, :, :, :], stddev[0, :, :, :]

	# img_total_max = img.max()
	for i in range(nframe):
		# print("frame: ", i)
		input_img[i, :, :] = img2[:, :, i, slno]

		if i>=fno_LV and i<fno_LV+nframe_metric:
			input_img0 = img2[:, :, i, slno]
			# if img_total_max < 256:
			# 	print('frame %d, img_total_max < 256. scale up for unet' % i)
			# 	input_img0 = input_img0 / 255.0 * 1200
			input_img0 = input_img0/input_img0.max()
			# MC dropout을 여기에 쓰면 좋을 듯 싶다.
			# randomness가 들어가므로 매 번 실행할 때마다 endo_frame, epic_frame이 다르게 나오는 경우가 있다.
			# 따라서 Dice score도 매번 다르게 나올 수 있음.
			mean, stddev = monte_carlo_predict(sess, input_img0, niter_mc)
			mean_endo[i, :, :] = mean[:, :, 0]; 			mean_epic[i, :, :] = mean[:, :, 1]
			sdev_endo[i, :, :] = stddev[:, :, 0]; 			sdev_epic[i, :, :] = stddev[:, :, 1]
			a = sdev_endo[i, :, :]; b = np.squeeze(a); nx, ny = b.shape; c = sdev_epic[i, :, :]; d = np.squeeze(c)
			e = mean_endo[i, :, :]; f = np.squeeze(e); g = mean_epic[i, :, :]; h = np.squeeze(g)
			sum_std_endo = np.sum(b.reshape((nx*ny, 1))); sum_std_epic = np.sum(d.reshape((nx * ny, 1)))
			sum_mean_endo = np.sum(f.reshape((nx*ny, 1))); sum_mean_epic=np.sum(h.reshape((nx*ny, 1)))
			# ind1 = np.where(b > thresh); 			ind2 = np.where(d > thresh)
			# n_std_high1.append(len(ind1[0])); 		n_std_high2.append(len(ind2[0]))
			sum_std_fn_endo.append(sum_std_endo/sum_mean_endo)
			sum_std_fn_epic.append(sum_std_epic/sum_mean_epic)
			# sum_mean_fn_endo.append(sum_mean_endo); sum_mean_fn_epic.append(sum_mean_epic)

			# print('frame = %d, endo std > thresh = %d, epi_std > thresh = %d' % (i, len(ind1[0]), len(ind2[0])))
			print('frame=%d, sum_std_endo=%4.2f, sum_std_epic=%4.2f, sum_mean_endo=%4.2f, sum_mean_epic=%4.2f' % (i, sum_std_endo, sum_std_epic,sum_mean_endo, sum_mean_epic))

		else:
			# n_std_high1.append(1e3); 			n_std_high2.append(1e3)
			sum_std_fn_endo.append(1e3); 		sum_std_fn_epic.append(1e3)
			# sum_mean_fn_endo.append(0.1); sum_mean_fn_epic.append(0.1)

	# print('sum_std: frame for endo = %d, SSD = %5.3f' % (np.argmin(np.asarray(sum_std_fn_endo)), min(np.asarray(sum_std_fn_endo))))
	# print('sum_std: frame for epic = %d, SSD = %5.3f' % (np.argmin(np.asarray(sum_std_fn_epic)), min(np.asarray(sum_std_fn_epic))))
	# frame_endo = np.argmin(np.asarray(n_std_high1)); 	frame_epic = np.argmin(np.asarray(n_std_high2))
	frame_endo_sstd = np.argmin(np.asarray(sum_std_fn_endo))
	frame_epic_sstd = np.argmin(np.asarray(sum_std_fn_epic))
	# nframe = input_img.shape[0]
	# n_std_endo_arr = np.asarray(n_std_high1)
	# ind_not_foi = np.where(n_std_endo_arr == 1e3)[0]
	# n_std_endo_arr[n_std_endo_arr == 1e3] = 0
	# n_std_epic_arr = np.asarray(n_std_high2)
	# n_std_epic_arr[n_std_epic_arr == 1e3] = 0

	sum_std_fn_endo_arr = np.asarray(sum_std_fn_endo)
	sum_std_fn_endo_arr[sum_std_fn_endo_arr == 1e3] = 0
	sum_std_fn_epic_arr = np.asarray(sum_std_fn_epic)
	sum_std_fn_epic_arr[sum_std_fn_epic_arr == 1e3] = 0

	clear_layers_name()
	reset_default_graph()

	with tf.device(device_name):
		t_image = tf.placeholder('float32', [1, input_img.shape[1], input_img.shape[2], 1], name='input_image')
		net = u_net(t_image, is_train=False, apply_dropout=False, reuse=False, n_out=2)

	def predict(sess, image):
		out = sess.run(net.outputs, {t_image: image[np.newaxis, :, :, np.newaxis]})
		return out[0, :, :, :]

	saver = tf.train.Saver()
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
	config = tf.ConfigProto(gpu_options=gpu_options)
	config.gpu_options.allow_growth = True
	sess = tf.InteractiveSession(config=config)
	saver.restore(sess, unet_fname)

	''' method 2. sum of std map '''
	frames_list = np.array((frame_endo_sstd, frame_epic_sstd)).tolist()

	for j, fno in enumerate(frames_list):
		img00 = input_img[fno, :, :]
		img00 = img00/img00.max()
		out = predict(sess, img00)
		if j == 0:
			mask_endo = out[:, :, 0]
		elif j == 1:
			mask_epic = out[:, :, 1]

	mask_epic_bin = np.zeros(mask_epic.shape)
	mask_epic_bin[mask_epic < 0.5] = 0
	mask_epic_bin[mask_epic >= 0.5] = 1

	mask_endo_bin = np.zeros(mask_endo.shape)
	mask_endo_bin[mask_endo < 0.5] = 0
	mask_endo_bin[mask_endo >= 0.5] = 1

	''' u-net binary mask에서 불필요한 island 없애기 '''
	# from scipy.ndimage.measurements import label
	# from skimage.measure import regionprops

	''' epicardium에 대해 수행 '''
	labeled_arr, nfeature = label(mask_epic_bin)
	# print(labeled_arr.shape)
	# print('no of labels = ', nfeature)
	props = regionprops(labeled_arr)
	# print(len(props))
	# get centroid of second object
	# centroid = props[0].centroid
	# get eccentricity of first object
	# ecc = props[0].eccentricity
	# print(centroid)

	max_area = -1000
	max_ind = 0
	for jj in range(nfeature):
		area = props[jj].filled_area
		# print('region %d, area=%5d'% (jj, area))
		if area > max_area:
			max_area = area
			max_ind = jj

	mask_epic2 = np.zeros(mask_epic.shape)
	mask_epic2[labeled_arr == max_ind + 1] = 1

	''' endo에 대해 수행 '''
	labeled_arr, nfeature = label(mask_endo_bin)
	# print('no of labels = ', nfeature)
	props = regionprops(labeled_arr)
	max_area = -1000
	max_ind = 0
	for jj in range(nfeature):
		area = props[jj].filled_area
		# print('region %d, area=%5d'% (jj, area))
		if area > max_area:
			max_area = area
			max_ind = jj
	mask_endo2 = np.zeros(mask_endo.shape)
	mask_endo2[labeled_arr == max_ind + 1] = 1
	mask_myo_mcdo = mask_epic2 - mask_endo2

	clear_layers_name()
	reset_default_graph()

	# mask_endo 내에 있는 복셀들의 TIC을 그리고, peak인 프레임을 fno_LV로 하면 좋겠다.
	ind2 = np.where(mask_endo2 == 1)
	tic_LV = np.zeros(nframe)
	for tt in range(nframe):
		img_ = img2[:, :, tt, slno]
		s1 = img_[ind2]
		m1 = np.mean(s1)
		tic_LV[tt] = m1
	fno_LV = np.argmax(tic_LV)

	if False:
		plt.figure()
		plt.plot(tic_LV)
		plt.title(f'fno_LV={fno_LV}')
		plt.show()

	return mask_epic2, mask_endo2, mask_myo_mcdo, input_img, img2, fno_RV, fno_LV, indRVcavity, ind_fat


def segment_myo_regional(mask_myo_mc, LVc_, RVi_, slno):
	'''
	LVc, RVi 점이 주어지면 이를 바탕으로 각도를 6등분한다.
	'''

	myosegment = np.zeros(mask_myo_mc.shape)
	_, ny = mask_myo_mc.shape

	if slno == 0 or slno==1 or slno==2:
		nseg = 6
	elif slno >=3:
		nseg = 4

	# mask = np.zeros(mask_myo_mc))

	zLVc = complex(LVc_[0], LVc_[1])
	zRVi = complex(RVi_[0], RVi_[1])
	z0 = zRVi - zLVc
	# print('zLVc, ', zLVc)
	# print('zRVi, ', zRVi)

	x1 = [LVc_]
	x2 = [RVi_]
	line1 = x1+x2
	# print('line1, ', line1)

	line_list = []
	line_list.append(line1)
	for ind in range(1, nseg):
		z1 = z0 * np.exp(-1j * 2 * np.pi * ind / nseg)
		zp1 = z1 + zLVc
		xp1 = [(zp1.real, zp1.imag)]
		line_list.append(x1 + xp1)

	line_ref = line_list[nseg-1]
	zref_pos = complex(line_ref[1][0], line_ref[1][1])
	zref = zref_pos - zLVc

	ind_myo = get_index_fromMask(mask_myo_mc)
	set_myo = set(ind_myo)

	# print('zref, ', zref)

	if nseg==6:
		for j in set_myo:
			zvox_pos = complex(j % ny, int(j / ny))
			zvox = zvox_pos - zLVc

			angle = cmath.phase(zref / zvox) * 180 / np.pi  # in degree. I did zref/zvox to correct the segment order
			if angle < 0:
				angle += 360
			if 0 < angle < 60:
				myosegment[int(j / ny), j % ny] = 1.0
			elif 60 < angle < 120:
				myosegment[int(j / ny), j % ny] = 2.0
			elif 120 < angle < 180:
				myosegment[int(j / ny), j % ny] = 3.0
			elif 180 < angle < 240:
				myosegment[int(j / ny), j % ny] = 4.0
			elif 240 < angle < 300:
				myosegment[int(j / ny), j % ny] = 5.0
			else:
				myosegment[int(j / ny), j % ny] = 6.0

	elif nseg==4:
		print('todo later.')

	return myosegment, nseg


def get_Xy_param(pkl_fname):

	with open(pkl_fname, 'rb') as f:
		x = pickle.load(f)

	r_LVc = x['r_LVc']
	lsize_LVc = x['lsize_LVc']
	r_RVa = x['r_RVa']
	lsize_RVa = x['lsize_RVa']

	Xy_param = {}
	Xy_param['r_LVc'] = r_LVc
	Xy_param['lsize_LVc'] = lsize_LVc
	Xy_param['r_RVa'] = r_RVa
	Xy_param['lsize_RVa'] = lsize_RVa

	return Xy_param

data_dir = r'data'

res_dir_landmark = os.path.join(os.getcwd(), 'dat/landmark')
res_dir_myoseg   = os.path.join(os.getcwd(), 'dat/myoseg')

slice_index = 0
device_name = '/gpu:1'
nrow_des, ncol_des = 256, 216

# print('MPRI auto dicom')

clf_LVc = joblib.load('landmark_model/model_rf_LVc.pkl')
clf_RVi_stress = joblib.load('landmark_model/model_rf_RVi_stressPERF_n=72_3.pkl')
Xy_param_stress = {}

clf_RVi_rest = joblib.load('../landmark/pkl_model/model_rf_RVi_restPERF_n=63_6.pkl')
Xy_param_rest = get_Xy_param('../landmark/pkl_Xy_ML/train_landmark_restperf_RVi_n=63_5.pkl')

disp_figure_flag = True


for jj, subjno in enumerate(subj_dir):

	subj_directory = os.path.join(data_dir, subjno)

	_, id = os.path.split(subj_directory)
	print('\n jj = %d, id = %s' % (jj, id))

	perf_stress = read_dcm(subj_directory, stress_flag=True)
	perf_rest   = read_dcm(subj_directory, stress_flag=False)

	img_stress = perf_stress['images']
	img_rest   = perf_rest['images']

	[nrow, ncol, nframe, nslice] = img_stress.shape

	if img_stress.shape[0] < img_stress.shape[1]:
		# rot90 to all frames. stress and rest
		img_stress_ = deepcopy(img_stress)
		nx, ny, nframe, nslice = img_stress_.shape
		img_rest_ = deepcopy(img_rest)

		img_stress2 = np.zeros((ny, nx, nframe, nslice))
		img_rest2 = np.zeros((ny, nx, nframe, nslice))

		for kk in range(nslice):
			for jj in range(nframe):
				tmp = img_stress_[:, :, jj, kk]
				img_stress2[:, :, jj, kk] = np.rot90(tmp)
				tmp2 = img_rest_[:, :, jj, kk]
				img_rest2[:, :, jj, kk] = np.rot90(tmp2)

		img_stress = img_stress2
		img_rest = img_rest2
		print('\t Row = %d, Column = %d' % (img_stress.shape[0], img_stress.shape[1]))

	img_stress_nomask = deepcopy(img_stress)
	img_rest_nomask   = deepcopy(img_rest)

	# gen_video(img_stress, img_rest, id)

	if 'MOCO' in perf_stress['SeriesDescription']:
		print('dicom already had MOCO. No need for registration')
	else:
		print('dicom had no MOCO. Perform motion correction in every frame!')

		''' perform registration '''

		''' stress '''

		''' rest '''

	st = time.time()

	nrow_orig, ncol_orig, _, _ = img_stress.shape
	mask_erode_flag = False
	mpri_list = []
	# mpri_map_odim = np.zeros((nrow, ncol, nslice))
	mpri_map_odim = np.zeros((nrow_orig, ncol_orig, nslice))

	for count, slno in enumerate(slno_list):  # basal, mid, apical

		for type in ['stress', 'rest']:
			print(f'\t\tind={jj}, slno={slno}, type={type}')
			if type == 'stress':
				# myocardial segmentation using mc dropout.
				st_stress_segment = time.time()
				mask_epic, mask_endo, mask_myo_mcdo, input_img, img, fno_RV, fno_LV, indRVcavity, ind_fat = \
					segment_myo(img_stress, slno, device_name, id, type)
				print(' elapsed time [stress MC segment] = %4.2f (s)' % (time.time() - st_stress_segment))
				# print(input_img.shape, img.shape)
				dir_ = os.path.join(res_dir_myoseg, 'stress')
				fname1 = 'myoseg_' + subjno + '_stress.npz'
				np.savez(os.path.join(dir_, fname1), input_img=input_img, fno_LV=fno_LV, mask_epic=mask_epic, mask_endo=mask_endo)

			elif type == 'rest':
				st_rest_segment = time.time()
				mask_epic, mask_endo, mask_myo_mcdo, input_img, img, fno_RV, fno_LV, indRVcavity, ind_fat = \
					segment_myo(img_rest, slno, device_name, id, type)
				print(' elapsed time [rest MC segment]   = %4.2f (s)' % (time.time() - st_rest_segment))
				dir_ = os.path.join(res_dir_myoseg, 'rest')
				fname1 = 'myoseg_' + subjno + '_rest.npz'
				np.savez(os.path.join(dir_, fname1), input_img=input_img, fno_LV=fno_LV, mask_epic=mask_epic, mask_endo=mask_endo)

			# convert to original dimension
			mask_endo_orig = np.zeros((nrow_orig, ncol_orig))
			mask_endo_orig[int((nrow_orig-nrow_des)/2):int((nrow_orig+nrow_des)/2),
							int((ncol_orig-ncol_des)/2):int((ncol_orig+ncol_des)/2)] = mask_endo

			if type == 'stress':
				# (320 x 276)
				LVc, RVi = detect_landmark(img_stress, slno, fno_RV, fno_LV, clf_LVc, clf_RVi_stress, Xy_param_stress, indRVcavity, ind_fat, mask_endo_orig)
				dir_ = os.path.join(res_dir_landmark, 'stress')
				fname1 = 'landmark_' + subjno + '_stress.npz'
				pixelspacing = perf_stress['PixelSpacing']
				np.savez(os.path.join(dir_, fname1), img_stress=img_stress_nomask, slno=slno, fno_RV=fno_RV, fno_LV=fno_LV, LVc=LVc, RVi=RVi, pixelspacing=pixelspacing)

			elif type == 'rest':
				LVc, RVi = detect_landmark(img_rest_nomask, slno, fno_RV, fno_LV, clf_LVc, clf_RVi_rest, Xy_param_rest, indRVcavity, ind_fat, mask_endo_orig)
				dir_ = os.path.join(res_dir_landmark, 'rest')
				fname1 = 'landmark_' + subjno + '_rest.npz'
				pixelspacing = perf_rest['PixelSpacing']
				np.savez(os.path.join(dir_, fname1), img_rest=img_rest_nomask, slno=slno, fno_RV=fno_RV, fno_LV=fno_LV, LVc=LVc, RVi=RVi, pixelspacing=pixelspacing)

			LVc_ = LVc.copy(); LVc_[0] = LVc[0] - (ncol_orig/2 - ncol_des/2); LVc_[1] = LVc[1] - (nrow_orig/2 - nrow_des/2)
			RVi_ = RVi.copy(); RVi_[0] = RVi[0] - (ncol_orig/2 - ncol_des/2); RVi_[1] = RVi[1] - (nrow_orig/2 - nrow_des/2)

			myosegment, nseg = segment_myo_regional(mask_myo_mcdo, LVc_, RVi_, slno)

			# regional calculation of upslope.
			upslope_map, upslope_myo, sig_LV_filt, xLV, yLV, sig_fit_myo, xmyo, ymyo = upslope_myo_segmental(img, slno, mask_endo, myosegment, nseg, fno_RV, fno_LV)
			mask_seg_list = []

			fit1 = np.polyfit(xLV, yLV, 1)
			fit1_fn = np.poly1d(fit1)
			# print('xmyo ', xmyo)
			# print('ymyo[0, :] ', ymyo[0,:])

			plt.figure('upslope ' + type)
			plt.subplot(231)
			plt.imshow(input_img[fno_LV, :, :], cmap='gray')
			plt.contour(mask_endo, [0.5], colors='r')
			plt.contour(mask_epic, [0.5], colors='r')
			# plt.scatter(LVc_[0], LVc_[1], marker='o', c='r', s=5)
			# plt.scatter(RVi_[0], RVi_[1], marker='o', c='y', s=5)

			plt.subplot(232)
			plt.imshow(mask_myo_mcdo, cmap='gray')

			plt.subplot(233)
			plt.plot(sig_LV_filt, 'r-', label='LV')
			plt.scatter(xLV, sig_LV_filt[xLV], marker='o', c='r')
			plt.plot(xLV, fit1_fn(xLV), 'k--', label='LV slope')
			plt.plot(sig_fit_myo[0, :], 'b-', label='myo')
			plt.scatter(xmyo, sig_fit_myo[0, xmyo], marker='o', c='b')
			# plt.plot(xmyo, fit2_fn_myo(xmyo), 'm--', label='myo slope')
			plt.legend()
			plt.title('LV, myo enhancement')

			plt.subplot(234)
			plt.imshow(upslope_map, cmap='jet', clim=[0, 0.4])
			plt.scatter(LVc_[0], LVc_[1], marker='+', c='r', s=5)
			plt.scatter(RVi_[0], RVi_[1], marker='+', c='b', s=5)
			plt.title('upslope map, '+type)

			plt.subplot(235)
			plt.imshow(input_img[fno_LV, :, :], cmap='gray')
			colorset = ['r', 'g', 'b', 'y', 'c', 'm']
			for jj in range(nseg):
				mask = np.zeros(myosegment.shape)
				mask[myosegment == jj+1] = 1
				if mask_erode_flag:
					diameter = 4; sel = morphology.disk(radius=int(diameter / 2))
					mask_erd = morphology.binary_erosion(mask, selem=sel)
				else:
					mask_erd = mask
				mask_seg_list.append(mask_erd)
				plt.contour(mask_erd, [0.5], colors=colorset[jj])
			for jj in range(nseg):
				mask = mask_seg_list[jj]
				ind = np.where(mask == 1)
				upslope_val = upslope_map[ind]
				print(' id=%s, slno=%d, %s, jj=%d, mean upslope = %4.3f' % (id, slno, type, jj, np.mean(upslope_val)))

			plt.subplot(236)
			plt.imshow(input_img[fno_LV, :, :], cmap='gray')
			# plt.contour(mask_endo, [0.5], colors='r')
			# plt.contour(mask_epic, [0.5], colors='r')
			plt.scatter(LVc_[0], LVc_[1], marker='s', c='b', s=13)
			plt.scatter(RVi_[0], RVi_[1], marker='o', c='r', s=13)

			# plt.show()

			plt.figure('segment-wise upslope - ' + type)
			for segno in range(nseg):
				slope, intercept, _, _, _ = scipy.stats.linregress(xmyo, ymyo[segno, :])

				fit2 = np.array([slope, intercept])
				fit2_fn_myo = np.poly1d(fit2)

				plt.subplot(2, 3, segno+1)
				plt.plot(sig_LV_filt, 'r-', label='LV')
				plt.scatter(xLV, sig_LV_filt[xLV], marker='o', c='r')
				plt.plot(xLV, fit1_fn(xLV), 'k--', label='LV slope')
				plt.plot(sig_fit_myo[segno, :], 'b-', label='myo')
				plt.scatter(xmyo, sig_fit_myo[segno, xmyo], marker='o', c='b')
				plt.plot(xmyo, fit2_fn_myo(xmyo), 'm--', label='myo slope')
				plt.legend()
				plt.title('%s, seg no = %d, upslope = %4.3f' % (type, segno, upslope_myo[segno]))

			if disp_figure_flag:
				plt.show()

			if type == 'stress':
				mask_seg_stress_list = deepcopy(mask_seg_list)
				upslope_map_stress = deepcopy(upslope_map)
			elif type == 'rest':
				mask_seg_rest_list = deepcopy(mask_seg_list)
				upslope_map_rest = deepcopy(upslope_map)

		print('slice no. = %d, MPRI' % slno)

		mpri_map2d = np.zeros((nrow_des, ncol_des))

		for jj in range(nseg):
			mask_stress = mask_seg_stress_list[jj]
			ind1 = np.where(mask_stress==1)
			upslope_stress_val = upslope_map_stress[ind1]

			mask_rest = mask_seg_rest_list[jj]
			ind2 = np.where(mask_rest==1)
			upslope_rest_val = upslope_map_rest[ind2]

			mpri = np.mean(upslope_stress_val)/np.mean(upslope_rest_val)
			print(' id=%s, slno=%d, jj=%d, MPRI = %4.3f' % (id, slno, jj, mpri))
			mpri_list.append(mpri)

			mpri_map2d[ind2] = mpri

		mpri_map_odim[int((nrow_orig-nrow_des)/2):int((nrow_orig+nrow_des)/2),int((ncol_orig-ncol_des)/2):int((ncol_orig+ncol_des)/2), slno] = mpri_map2d

	# img_overlay = img_rest_nomask[:,:, 28, 0]
	img_overlay = img_rest_nomask[:, :, 26, 0]
	# img_overlay = img_rest_nomask[:,:,28, 0]

	mask_mpri = mpri_map_odim[:, :, 0]
	# print(img_overlay.shape, mask_mpri.shape)

	if disp_figure_flag:
		mask_mpri_masked = np.ma.masked_where(mask_mpri == 0, mask_mpri)

		# MPRI overlaid to stress perf image~!!
		fig, ax = plt.subplots(1,1, figsize=(3,4))
		ax.imshow(img_overlay, cmap='gray', clim=((0, 448)))
		cax = ax.imshow(mask_mpri_masked, cmap='jet', clim=((0.0, 2.0)))
		fig.colorbar(cax, ticks=[0, 1.0, 2.0])
		plt.show()

	if False:
		''' Save xlsx '''
		if slno<=1 and count==0:
			df1 = pd.DataFrame([mpri_list], index=[id], columns=['mpri1', 'mpri2', 'mpri3', 'mpri4', 'mpri5', 'mpri6'])
			df1.to_excel('mpri_basal_' + id + '.xlsx')
		elif count==2:
			df1 = pd.DataFrame([mpri_list], index=[id], columns=['mpri1', 'mpri2', 'mpri3', 'mpri4', 'mpri5', 'mpri6', 'mpri7', 'mpri8', 'mpri9', 'mpri10', 'mpri11', 'mpri12', 'mpri13', 'mpri14', 'mpri15', 'mpri16'])
			df1.to_excel('mpri_3slices_' + id + '.xlsx')

	# plt.show()

	''' clear variables '''
	del img_stress_nomask
	del img_rest_nomask

	gc.collect()

	print('total elapsed time = %4.2f (s)' % (time.time() - st))


if disp_figure_flag:

	plt.show()
