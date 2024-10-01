

import os
import numpy as np
import os.path
import skimage
import pandas as pd
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import se3lib

import utils
import net
from config import Config

import urso
import speed
import torch

from scipy.spatial.transform import Rotation as R

# Ensure that your model and tensors are on the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device pose esti {device}")


# Models directory (where weights are stored)
MODEL_DIR = os.path.abspath("./models")
DEFAULT_LOGS_DIR = os.path.join(MODEL_DIR, "logs")
save_dir = "datasets/results/"

# Dataset directory
DATA_DIR = os.path.abspath("UrsoNet-master/datasets/")

# Path to trained weights file of Mask-RCNN on COCO
COCO_WEIGHTS_PATH = os.path.join(MODEL_DIR, "mask_rcnn_coco.h5")

OrientationParamOptions = ['quaternion', 'euler_angles', 'angle_axis']

def fit_GMM_to_orientation(q_map, pmf, nr_iterations, var, nr_max_modes=4):
    ''' Fits multiple quaternions to a PMF using Expectation Maximization'''

    nr_total_bins = len(pmf)
    scores = []

    # Sorting bins per probability
    pmf_sorted_indices = pmf.argsort()[::-1]

    for N in range(1, nr_max_modes):

        # 1. Initialize Gaussians
        Q_mean = np.zeros((N,4), np.float32)
        Q_var = np.ones(N, np.float32)*var
        priors = np.ones(N, np.float32)/N

        # Initialize Gaussian means by picking up the strongest bins
        check_q_mask = np.zeros_like(pmf)>0

        ptr = 0
        for k in range(N):

            # Select bin
            for i in range(ptr, nr_total_bins):
                if not check_q_mask[i]:
                    check_q_mask[i] = True
                    q_max = q_map[pmf_sorted_indices[i], :]
                    Q_mean[k, :] = q_max
                    ptr = i + 1
                    break

            # Mask out neighbours
            for i in range(nr_total_bins):
                q_i = q_map[pmf_sorted_indices[i], :]
                if not check_q_mask[i]:
                    #d_i = (1 - np.sum(q_i * q_max)) ** 2
                    d_i = (se3lib.angle_between_quats(q_i, q_max) / 180) ** 2
                    if d_i < 9 * var:
                        check_q_mask[i] = 1


        # 2. Expectation Maximization loop
        for it in range(nr_iterations):

            # Expectation step

            # Normalized angular distance
            Distances = np.asarray(se3lib.angle_between_quats(q_map, Q_mean))/180

            # Compute p(X|Theta)
            eps = 1e-18
            p_X_given_models = eps + np.divide(np.exp(np.divide(-Distances ** 2, 2.0 * Q_var)),
                                                 np.sqrt(2.0 * np.pi * Q_var))

            # Compute p(Theta|X) by applying Bayes rule
            # Get marginal likelihood
            p_X_given_models_times_priors = p_X_given_models*priors
            p_X = np.sum(p_X_given_models_times_priors, axis=1)
            p_models_given_X = p_X_given_models_times_priors/p_X[:,np.newaxis]

            # Maximization step

            # Compute weights
            W = p_models_given_X * pmf[:, np.newaxis]
            Z = np.sum(W, axis=0)
            W_n = W / Z

            # Compute average quaternions
            for k in range(N):

                q_mean_k, _ = se3lib.quat_weighted_avg(q_map, W_n[:, k])
                Q_mean[k, :] = q_mean_k
                Q_var[k] = 0
                Distances = np.asarray(se3lib.angle_between_quats(q_map,q_mean_k)/180)**2
                for i in range(nr_total_bins):
                    Q_var[k] += W_n[i, k] * Distances[i]

            # print('New mixture means:\n', Q_mean)
            # print('New mixture priors:\n', priors)
            # print('New mixture var:\n', Q_var)
            # print('\n')

            # Compute priors
            priors = Z

            if N == 1 and it == 1:
                break

        # Check model likelihood by reusing last iteration state
        score = np.sum(pmf * np.log(p_X))

        if len(scores)==0 or score > scores[-1]+0.005:
            # Update best model
            Q_mean_best = Q_mean
            Q_var_best = Q_var
            Q_priors_best = priors
            scores.append(score)
        else:
            # Stop model searching to return last state
            break

    # TODO: Sort by likelihood
    sorting_indices = Q_priors_best.argsort()[::-1]

    Q_mean_best = Q_mean_best[sorting_indices]
    Q_priors_best = Q_priors_best[sorting_indices]
    Q_var_best = Q_var_best[sorting_indices]

    print('Q priors:',Q_priors_best)
    print('Q :', Q_mean_best)
    print('Scores:', scores)

    return Q_mean_best, Q_var_best, Q_priors_best, scores

def evaluate_image(model, dataset, image_id):

    # Load pose in all formats
    loc_gt = dataset.load_location(image_id)
    q_gt = dataset.load_quaternion(image_id)
    image = dataset.load_image(image_id)
    I, I_meta, loc_encoded_gt, ori_encoded_gt = \
        net.load_image_gt(dataset, model.config, image_id)

    results = model.detect([image], verbose=1)

    # Retrieve location
    if model.config.REGRESS_LOC:
        loc_est = results[0]['loc']
    else:
        loc_pmf = utils.stable_softmax(results[0]['loc'])

        # Compute location mean according to first moment
        loc_est = np.asmatrix(loc_pmf) * np.asmatrix(dataset.histogram_3D_map)

        # Compute loc encoding error
        loc_decoded_gt = np.asmatrix(loc_encoded_gt) * np.asmatrix(dataset.histogram_3D_map)
        loc_encoded_err = np.linalg.norm(loc_decoded_gt - loc_gt)

    # Retrieve orientation
    if model.config.REGRESS_ORI:

        if model.config.ORIENTATION_PARAM == 'quaternion':
            q_est = results[0]['ori']
        elif model.config.ORIENTATION_PARAM == 'euler_angles':
            q_est = se3lib.SO32quat(
                se3lib.euler2SO3_left(results[0]['ori'][0], results[0]['ori'][1], results[0]['ori'][2]))
        elif model.config.ORIENTATION_PARAM == 'angle_axis':
            theta = np.linalg.norm(results[0]['ori'])
            if theta < 1e-6:
                v = [0, 0, 0]
            else:
                v = results[0]['ori'] / theta
            q_est = se3lib.angleaxis2quat(v, theta)
    else:
        ori_pmf = utils.stable_softmax(results[0]['ori'])

        # Compute mean quaternion
        q_est, q_est_cov = se3lib.quat_weighted_avg(dataset.ori_histogram_map, ori_pmf)

        # Compute encoded error
        q_encoded_gt, _ = se3lib.quat_weighted_avg(dataset.ori_histogram_map, ori_encoded_gt)
        ori_encoded_err = 2 * np.arccos(
            np.abs(np.asmatrix(q_encoded_gt) * np.asmatrix(q_gt).transpose())) * 180 / np.pi

    # Compute errors
    angular_err = 2 * np.arccos(np.abs(np.asmatrix(q_est) * np.asmatrix(q_gt).transpose()))
    # angular_err_in_deg = angular_err* 180 / np.pi
    loc_err = np.linalg.norm(loc_est - loc_gt)
    loc_rel_err = loc_err / np.linalg.norm(loc_gt)

    # Compute ESA score
    esa_score = loc_rel_err + angular_err

    return loc_err, angular_err, loc_rel_err, esa_score


def evaluate(model, dataset):
    """ Evaluates model on all dataset images. Assumes all images have corresponding pose labels.
    """

    loc_err_acc = []
    loc_encoded_err_acc = []
    ori_err_acc = []
    ori_encoded_err_acc = []
    distances_acc = []
    esa_scores_acc = []

    # Variance used only for prob. orientation estimation
    delta = model.config.BETA / model.config.ORI_BINS_PER_DIM
    var = delta ** 2 / 12

    for image_id in dataset.image_ids:

        print('Image ID:', image_id)

        # Load pose in all formats
        loc_gt = dataset.load_location(image_id)
        q_gt = dataset.load_quaternion(image_id)
        image = dataset.load_image(image_id)

        results = model.detect([image], verbose=1)

        if model.config.REGRESS_KEYPOINTS:
            # Experimental

            I, I_meta, loc_gt, k1_gt, k2_gt = \
                net.load_image_gt(dataset, model.config, image_id)

            loc_est = results[0]['loc']
            k1_est = results[0]['k1']
            k2_est = results[0]['k2']

            # Prepare keypoint matches
            # TODO: take scale into account and get rid of magic numbers
            P1 = np.zeros((3, 3))
            P1[2,0] = 3.0
            P1[1,1] = 3.0

            P2 = np.zeros((3, 3))
            P2[:, 0] = k1_est
            P2[:, 1] = k2_est
            P2[:, 2] = loc_est

            t, R = se3lib.pose_3Dto3D(np.asmatrix(P1),np.asmatrix(P2))
            q_est = se3lib.SO32quat(R.T)

        else:

            I, I_meta, loc_encoded_gt, ori_encoded_gt = \
                net.load_image_gt(dataset, model.config, image_id)

            # Retrieve location
            if model.config.REGRESS_LOC:
                loc_est = results[0]['loc']
            else:
                loc_pmf = utils.stable_softmax(results[0]['loc'])

                # Compute location mean according to first moment
                loc_est = np.asmatrix(loc_pmf) * np.asmatrix(dataset.histogram_3D_map)

                # Compute loc encoding error
                loc_decoded_gt = np.asmatrix(loc_encoded_gt) * np.asmatrix(dataset.histogram_3D_map)
                loc_encoded_err = np.linalg.norm(loc_decoded_gt - loc_gt)
                loc_encoded_err_acc.append(loc_encoded_err)

            # Retrieve orientation
            if model.config.REGRESS_ORI:

                if model.config.ORIENTATION_PARAM == 'quaternion':
                    q_est = results[0]['ori']
                elif model.config.ORIENTATION_PARAM == 'euler_angles':
                    q_est = se3lib.SO32quat(se3lib.euler2SO3_left(results[0]['ori'][0], results[0]['ori'][1], results[0]['ori'][2]))
                elif model.config.ORIENTATION_PARAM == 'angle_axis':
                    theta = np.linalg.norm(results[0]['ori'])
                    if theta < 1e-6:
                        v = [0,0,0]
                    else:
                        v = results[0]['ori']/theta
                    q_est = se3lib.angleaxis2quat(v,theta)
            else:

                ori_pmf = utils.stable_softmax(results[0]['ori'])

                # Compute mean quaternion
                q_est, q_est_cov = se3lib.quat_weighted_avg(dataset.ori_histogram_map, ori_pmf)

                # Multimodal estimation
                # Uncomment this block to try the EM framework
                # nr_EM_iterations = 5
                # Q_mean, Q_var, Q_priors, model_scores = fit_GMM_to_orientation(dataset.ori_histogram_map, ori_pmf,
                #                                                                nr_EM_iterations, var)
                #
                # print('Err:', angular_err)
                # angular_err = 2*np.arccos(np.abs(np.asmatrix(Q_mean)*np.asmatrix(q_gt).transpose()))*180/np.pi
                #
                # # Select best of two
                # if len(angular_err) == 1 or angular_err[0]<angular_err[1]:
                #     q_est = Q_mean[0, :]
                # else:
                #     q_est = Q_mean[1, :]
                #
                # print('Err:',angular_err)

                # Compute encoded error
                q_encoded_gt, _ = se3lib.quat_weighted_avg(dataset.ori_histogram_map, ori_encoded_gt)
                ori_encoded_err = 2*np.arccos(np.abs(np.asmatrix(q_encoded_gt)*np.asmatrix(q_gt).transpose()))*180/np.pi
                ori_encoded_err_acc.append(ori_encoded_err)

        # 3. Angular error
        angular_err = 2*np.arccos(np.abs(np.asmatrix(q_est)*np.asmatrix(q_gt).transpose()))*180/np.pi
        ori_err_acc.append(angular_err.item(0))

        # 4. Loc error
        loc_err = np.linalg.norm(loc_est - loc_gt)
        loc_err_acc.append(loc_err)

        print('Loc Error: ', loc_err)
        print('Ori Error: ', angular_err)

        # Compute ESA score
        esa_score = loc_err/np.linalg.norm(loc_gt) + 2*np.arccos(np.abs(np.asmatrix(q_est)*np.asmatrix(q_gt).transpose()))
        esa_scores_acc.append(esa_score)

        # Store depth
        distances_acc.append(loc_gt[2])

    print('Mean est. location error: ', np.mean(loc_err_acc))
    print('Mean est. orientation error: ', np.mean(ori_err_acc))
    print('ESA score: ', np.mean(esa_scores_acc))
    print('Mean encoded location error: ', np.mean(loc_encoded_err_acc))

    # Dump results
    pd.DataFrame(np.asarray(ori_err_acc)).to_csv("ori_err.csv")
    pd.DataFrame(np.asarray(loc_err_acc)).to_csv("loc_err.csv")
    pd.DataFrame(np.asarray(distances_acc)).to_csv("dists_err.csv")


def detect_dataset(model, dataset, nr_images):
    """ Tests model on N random images of the dataset
     and shows the results.
    """

    # Variance used only for prob. orientation estimation
    delta = model.config.BETA / model.config.ORI_BINS_PER_DIM
    var = delta ** 2 / 12

    for i in range(nr_images):
        image_id = random.choice(dataset.image_ids)

        # Load pose in all formats
        loc_gt = dataset.load_location(image_id)
        q_gt = dataset.load_quaternion(image_id)
        I, I_meta, loc_encoded_gt, ori_encoded_gt = \
            net.load_image_gt(dataset, model.config, image_id)
        image_ori = dataset.load_image(image_id)

        info = dataset.image_info[image_id]

        # Run detection
        results = model.detect([image_ori], verbose=1)

        # Retrieve location
        if model.config.REGRESS_LOC:
            loc_est = results[0]['loc']
        else:
            loc_pmf = utils.stable_softmax(results[0]['loc'])

            # Compute location mean according to first moment
            loc_est = np.asmatrix(loc_pmf) * np.asmatrix(dataset.histogram_3D_map)

            # Compute loc encoding error
            loc_encoded_gt = np.asmatrix(loc_encoded_gt) * np.asmatrix(dataset.histogram_3D_map)
            loc_encoded_err = np.linalg.norm(loc_encoded_gt - loc_gt)

        # Retrieve orientation
        if model.config.REGRESS_ORI:

            if model.config.ORIENTATION_PARAM == 'quaternion':
                q_est = results[0]['ori']
            elif model.config.ORIENTATION_PARAM == 'euler_angles':
                q_est = se3lib.SO32quat(se3lib.euler2SO3_left(results[0]['ori'][0], results[0]['ori'][1], results[0]['ori'][2]))
            elif model.config.ORIENTATION_PARAM == 'angle_axis':
                theta = np.linalg.norm(results[0]['ori'])
                if theta < 1e-6:
                    v = [0,0,0]
                else:
                    v = results[0]['ori']/theta
                q_est = se3lib.angleaxis2quat(v,theta)
        else:
            ori_pmf = utils.stable_softmax(results[0]['ori'])

            # Compute mean quaternion
            q_est, q_est_cov = se3lib.quat_weighted_avg(dataset.ori_histogram_map, ori_pmf)

            # Multimodal estimation
            # Uncomment this block to try the EM framework
            # nr_EM_iterations = 5
            # Q_mean, Q_var, Q_priors, model_scores = fit_GMM_to_orientation(dataset.ori_histogram_map, ori_pmf, nr_EM_iterations, var)
            # print('Multimodal errors',2 * np.arccos(np.abs(np.asmatrix(Q_mean) * np.asmatrix(q_gt).transpose())) * 180 / np.pi)
            #
            # q_est_1 = Q_mean[0, :]
            # q_est_2 = Q_mean[1, :]
            # utils.polar_plot(q_est_1, q_est_2)

        # Compute Errors
        angular_err = 2 * np.arccos(np.abs(np.asmatrix(q_est) * np.asmatrix(q_gt).transpose())) * 180 / np.pi
        loc_err = np.linalg.norm(loc_est - loc_gt)

        print('GT location: ', loc_gt)
        print('Est location: ', loc_est)
        print('Processed Image:', info['path'])
        print('Est orientation: ', q_est)
        print('GT_orientation: ', q_gt)

        print('Location error: ', loc_err)
        print('Angular error: ', angular_err)

        # Visualize PMFs
        # if not model.config.REGRESS_ORI:

        #     nr_bins_per_dim = model.config.ORI_BINS_PER_DIM
        #     utils.visualize_weights(ori_encoded_gt,ori_pmf,nr_bins_per_dim)

        # Show image
        fig, (ax_1, ax_2) = plt.subplots(1, 2, figsize=(12, 8))
        #ax_1.imshow(image_ori)
        ax_1.set_xticks([])
        ax_1.set_yticks([])
        # ax_2.imshow(image_ori)
        ax_2.set_xticks([])
        ax_2.set_yticks([])

        height_ori = np.shape(image_ori)[0]
        width_ori = np.shape(image_ori)[1]

        # Recover focal lengths
        fx = dataset.camera.fx
        fy = dataset.camera.fy
        K = np.matrix([[fx, 0, width_ori / 2], [0, fy, height_ori / 2], [0, 0, 1]])

        # Speed labels expresses q_obj_cam whereas Urso labels expresses q_cam_obj
        if dataset.name == 'Speed':
            q_est = se3lib.quat_inv(q_est)
            q_gt = se3lib.quat_inv(q_gt)

        # utils.visualize_axes(ax_1, q_gt, loc_gt, K, 100)
        # utils.visualize_axes(ax_2, q_est, loc_est, K, 100)

        # Save the figure
        #fig.savefig(os.path.join(save_dir, f'detection_{image_id}_axes.png'))
        plt.close(fig)

        #utils.polar_plot(q_gt, q_est)
        #plt.savefig(os.path.join(save_dir, f'detection_{image_id}_polar.png'))
        plt.close()

        # Location overlap visualization
        fig, ax = plt.subplots()
        ax.imshow(image_ori)

        # Project 3D coords for visualization
        x_est = loc_est[0] / loc_est[2]
        y_est = loc_est[1] / loc_est[2]
        x_gt = loc_gt[0] / loc_gt[2]
        y_gt = loc_gt[1] / loc_gt[2]

        if not model.config.REGRESS_LOC:
            x_decoded_gt = loc_encoded_gt[0, 0] / loc_encoded_gt[0, 2]
            y_decoded_gt = loc_encoded_gt[0, 1] / loc_encoded_gt[0, 2]
            circ = Circle((x_decoded_gt * fx + width_ori / 2, height_ori / 2 + y_decoded_gt * fy), 7, facecolor='b', label='encoded')
            ax.add_patch(circ)

        circ_gt = Circle((x_gt * fx + width_ori / 2, height_ori / 2 + y_gt * fy), 15, facecolor='r', label='gt')
        ax.add_patch(circ_gt)
        circ = Circle((x_est * fx + width_ori / 2, height_ori / 2 + y_est * fy), 10, facecolor='g', label='pred')
        ax.add_patch(circ)
        ax.legend(loc='upper right', shadow=True, fontsize='x-small')

        # Save the figure
        #fig.savefig(os.path.join(save_dir, f'detection_{image_id}_locations.png'))
        plt.close(fig)


def train(model, dataset_train, dataset_val):
    """Train the model."""

    model.config.STEPS_PER_EPOCH = min(1000,int(len(dataset_train.image_ids)/model.config.BATCH_SIZE))

    # Write config to disk
    config_filename = 'config_' + str(model.epoch) + '.json'
    config_filepath = os.path.join(model.log_dir, config_filename)
    model.config.write_to_file(config_filepath)

    print("Training")
    model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=config.EPOCHS, layers='all')



# load test_poses_gt.csv
import re
data = []


test_pose_gt_path = DATA_DIR + '/Datasets_UE_Satellite/' + 'test_poses_gt.csv' 

with open(test_pose_gt_path, 'r') as file:
    header = file.readline().strip().split(",")
    for line in file.readlines():
        match = re.match(r"(Image_.*?\.png),(.*)", line.strip())
        if match:
            image_file = match.group(1)  # Image file name
            rest_of_data = match.group(2).split(",")  # The remaining data (x, y, z, etc.)
            data.append([image_file] + rest_of_data)  # Append to the list

# Define column names
columns = ["ImageFile", "x", "y", "z", "wx", "wy", "wz", "t"]

# Convert to DataFrame
test_pose_gt_df = pd.DataFrame(data, columns=columns)


def evaluate(model, dataset, output_file='datasets/results/predictions.txt', start_index=0, end_index=None, img_no=4):


    if end_index is None:
        end_index = len(dataset.image_ids)

    if start_index < 0 or end_index > len(dataset.image_ids):
        raise ValueError(f"start_index ({start_index}) or end_index ({end_index}) out of bounds.")

    loc_err_acc = []
    ori_err_acc = []
    distances_acc = [] 
    esa_scores_acc = []

    loc_est_map = {}

    all_image_results = []

    
    delta = model.config.BETA / model.config.ORI_BINS_PER_DIM
    var = delta ** 2 / 12

    write_header = not os.path.exists(output_file)

    # Process images in batches between start_index and end_index
    for batch_start in range(start_index, end_index, img_no):
        batch_end = min(batch_start + img_no, end_index)
        image_ids_batch = dataset.image_ids[batch_start:batch_end]

        for image_id in image_ids_batch:
            

            print(f'Processing Image ID: {image_id}')

            # Load pose in all formats
            loc_gt_tuple = tuple(dataset.load_location(image_id))
            q_gt = dataset.load_quaternion(image_id)
            image = dataset.load_image(image_id)

            results = model.detect([image], verbose=1)

            I, I_meta, loc_encoded_gt, ori_encoded_gt = \
                net.load_image_gt(dataset, model.config, image_id)

            # Retrieve location
            if model.config.REGRESS_LOC:
                loc_est = results[0]['loc']
            else:
                loc_pmf = utils.stable_softmax(results[0]['loc'])
                loc_est = np.asmatrix(loc_pmf) * np.asmatrix(dataset.histogram_3D_map)

            # Retrieve orientation
            if model.config.REGRESS_ORI:
                if model.config.ORIENTATION_PARAM == 'quaternion':
                    q_est = results[0]['ori']
                elif model.config.ORIENTATION_PARAM == 'euler_angles':
                    q_est = se3lib.SO32quat(se3lib.euler2SO3_left(
                        results[0]['ori'][0], results[0]['ori'][1], results[0]['ori'][2]))
                elif model.config.ORIENTATION_PARAM == 'angle_axis':
                    theta = np.linalg.norm(results[0]['ori'])
                    v = results[0]['ori'] / theta if theta >= 1e-6 else [0, 0, 0]
                    q_est = se3lib.angleaxis2quat(v, theta)
            else:
                ori_pmf = utils.stable_softmax(results[0]['ori'])
                q_est, q_est_cov = se3lib.quat_weighted_avg(dataset.ori_histogram_map, ori_pmf)

            # Calculate errors
            angular_err = 2 * np.arccos(np.abs(np.asmatrix(q_est) * np.asmatrix(q_gt).transpose())) * 180 / np.pi
            loc_err = np.linalg.norm(loc_est - loc_gt_tuple)

            #Load image details
            image_info = dataset.image_info[image_id]
            image_name = image_info['name']

            # Retrieve wx, wy, wz (ground truth orientation in Euler angles) from test_pose_gt_df
            euler_gt = test_pose_gt_df.loc[test_pose_gt_df['ImageFile'] == image_name, ['wx', 'wy', 'wz']].values[0]
            timestep = test_pose_gt_df.loc[test_pose_gt_df['ImageFile'] == image_name, 't'].values[0]

            # Convert quaternions (q_gt and q_est) to Euler angles
            euler_est = se3lib.quat2euler(q_est)

            # Save the result for this image
            all_image_results.append({
                'image_name': image_name,
                'timestep': timestep,
                'loc_gt': loc_gt_tuple,
                'q_gt_euler': euler_gt,
                'loc_est': loc_est,
                'q_est': euler_est,
                'loc_err': loc_err,
                'angular_err': angular_err
            })

            # Add the loc_est to the loc_est_map for averaging later
            if loc_gt_tuple not in loc_est_map:
                loc_est_map[loc_gt_tuple] = {'loc_est_sum': np.array(loc_est), 'count': 1}
            else:
                loc_est_map[loc_gt_tuple]['loc_est_sum'] += np.array(loc_est)
                loc_est_map[loc_gt_tuple]['count'] += 1

    # Compute average estimated locations for each unique loc_gt
    avg_loc_est_map = {}
    for loc_gt, data in loc_est_map.items():
        avg_loc_est_map[loc_gt] = data['loc_est_sum'] / data['count']

    # Open the output file to append predictions
    with open(output_file, 'a') as f:
        # Write the header row if the file does not exist
        if write_header:
            f.write("Image ID,timestep,x,y,z,ox,oy,oz,est x,est y,est z,est ox,est oy,est oz,avg est x,avg est y,avg est z,Loc Err,Angular Err\n")

        for result in all_image_results:
            loc_gt = result['loc_gt']
            avg_loc_est = avg_loc_est_map[tuple(loc_gt)]

            loc_gt_str = ', '.join(map(str, result['loc_gt']))
            q_gt_str = ', '.join(map(str, result['q_gt_euler']))
            loc_est_str = ', '.join(map(str, result['loc_est']))  # Actual loc_est
            avg_loc_est_str = ', '.join(map(str, avg_loc_est))  # Average loc_est
            q_est_str = ', '.join(map(str, result['q_est']))

            # Write the results for each image
            f.write(f"{result['image_name']},{result['timestep']},{loc_gt_str},{q_gt_str},{loc_est_str},{q_est_str},{avg_loc_est_str},{result['loc_err']},{result['angular_err']}\n")

    print(f"Predictions from index {start_index} to {end_index} written to {output_file}")


############################################################
#  Main
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("command", metavar="<command>", help="'train' or 'evaluate'")
    parser.add_argument('--backbone', required=False, default='resnet50',help='Backbone architecture')
    parser.add_argument('--dataset', required=True, help='Dataset name')
    parser.add_argument('--epochs', required=False, default=100, type=int, help='Number of epochs')
    parser.add_argument('--image_scale', required=False, default=1.0, type=float, help='Resize scale')
    parser.add_argument('--ori_weight', required=False, default=1.0, type=float, help='Loss weight')
    parser.add_argument('--loc_weight', required=False, default=1.0, type=float, help='Loss weight')
    parser.add_argument('--bottleneck', required=False, default=32, type=int, help='Bottleneck width')
    parser.add_argument('--branch_size', required=False, default=1024, type=int, help='Branch input size')
    parser.add_argument('--learn_rate', required=False, default=0.001, type=float, help='Learning rate')
    parser.add_argument('--batch_size', required=False, default=4, type=int, help='Number of images per GPU')
    parser.add_argument('--rot_aug', dest='rot_aug', action='store_true')
    parser.set_defaults(rot_aug=False)
    parser.add_argument('--rot_image_aug', dest='rot_image_aug', action='store_true')
    parser.set_defaults(rot_image_aug=False)
    parser.add_argument('--classify_ori', dest='regress_ori', action='store_false')
    parser.add_argument('--regress_ori', dest='regress_ori', action='store_true')
    parser.set_defaults(regress_ori=False)
    parser.add_argument('--classify_loc', dest='regress_loc', action='store_false')
    parser.add_argument('--regress_loc', dest='regress_loc', action='store_true')
    parser.set_defaults(regress_loc=True)
    parser.add_argument('--regress_keypoints', dest='regress_keypoints', action='store_true') # Experimental: Overides options above
    parser.set_defaults(regress_keypoints=False)
    parser.add_argument('--sim2real', dest='sim2real', action='store_true')
    parser.set_defaults(sim2real=False)
    parser.add_argument('--clr', dest='clr', action='store_true')
    parser.set_defaults(clr=False)
    parser.add_argument('--f16', dest='f16', action='store_true')
    parser.set_defaults(f16=False)
    parser.add_argument('--square_image', dest='square_image', action='store_true')
    parser.set_defaults(square_image=False)
    parser.add_argument('--ori_param', required=False, default='quaternion', help="'quaternion' 'euler_angles' 'angle_axis'")
    parser.add_argument('--ori_resolution', required=False, default=16, type=int, help="Number of bins assigned to each angle")
    parser.add_argument('--weights', required=True, help="Path to weights .h5 file or 'coco' or 'imagenet' for coco pre-trained weights")
    parser.add_argument('--logs', required=False, default=DEFAULT_LOGS_DIR, help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False, metavar="path or URL to image", help='Image to evaluate') 

    # new arguments

    # New arguments for batch processing with start and end index
    parser.add_argument('--start_index', required=False, type=int, default=0, help='Start index for image evaluation')
    parser.add_argument('--end_index', required=False, type=int, help='End index for image evaluation')
    parser.add_argument('--img_no', required=False, type=int, default=10, help='Batch size for evaluation')


    args = parser.parse_args()
    print("Command: ", args.command)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    assert args.ori_param in OrientationParamOptions

    # Set up configuration
    config = Config()
    config.ORIENTATION_PARAM = args.ori_param # only used in regression mode
    config.ORI_BINS_PER_DIM = args.ori_resolution # only used in classifcation mode
    config.NAME = args.dataset
    config.EPOCHS = args.epochs
    config.NR_DENSE_LAYERS = 1 # Number of fully connected layers used on top of the feature network
    config.LEARNING_RATE = args.learn_rate # 0.001
    config.BOTTLENECK_WIDTH = args.bottleneck
    config.BRANCH_SIZE = args.branch_size
    config.BACKBONE = args.backbone
    config.ROT_AUG = args.rot_aug
    config.F16 = args.f16
    config.SIM2REAL_AUG = args.sim2real
    config.CLR = args.clr
    config.ROT_IMAGE_AUG = args.rot_image_aug
    config.OPTIMIZER = "SGD"
    config.REGRESS_ORI = args.regress_ori
    config.REGRESS_LOC = args.regress_loc
    config.REGRESS_KEYPOINTS = args.regress_keypoints
    config.LOSS_WEIGHTS['loc_loss'] = args.loc_weight
    config.LOSS_WEIGHTS['ori_loss'] = args.ori_weight

    # Set up resizing & padding if needed
    if args.square_image:
        config.IMAGE_RESIZE_MODE = 'square'
    else:
        config.IMAGE_RESIZE_MODE = 'pad64'

    if args.dataset == "speed":
        width_original = speed.Camera.width
        height_original = speed.Camera.height
    else:
        width_original = urso.Camera.width
        height_original = urso.Camera.height

    config.IMAGE_MAX_DIM = round(width_original * args.image_scale)

    if config.IMAGE_MAX_DIM % 64 > 0:
        raise Exception("Scale problem. Image maximum dimension must be dividable by 2 at least 6 times.")

    # n.b: assumes height is less than width
    height_scaled = round(height_original * args.image_scale)
    if height_scaled % 64 > 0:
        config.IMAGE_MIN_DIM = height_scaled - height_scaled%64 + 64
    else:
        config.IMAGE_MIN_DIM = height_scaled

    # Uncomment this if the model is trained from scratch
    # if args.dataset == "speed":
    #     config.NR_IMAGE_CHANNELS = 1

    if args.command == "train":
        config.IMAGES_PER_GPU = args.batch_size
    else:
        config.IMAGES_PER_GPU = 1
    config.BATCH_SIZE = config.IMAGES_PER_GPU * config.GPU_COUNT

    config.update()
    config.display()


    test_pose_gt_path = os.path.join(DATA_DIR, args.dataset, "test_poses_gt.csv")


    # Create model
    if args.command == "train":
        model = net.UrsoNet(mode="training", config=config,
                             model_dir=args.logs)
    else:
        model = net.UrsoNet(mode="inference", config=config,
                             model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        _, weights_path = model.find_last()
    elif args.weights.lower() != "none":
        _, weights_path = model.get_last_checkpoint(args.weights)

    # Load weights
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, None, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    elif args.weights.lower() == "imagenet":
        model.load_weights(weights_path, None, by_name=True)
    elif args.weights.lower() != "none":
        model.load_weights(weights_path, weights_path, by_name=True)
        #model.load_weights(weights_path, weights_path, by_name=True, exclude=["ori_final"]) # tmp

    dataset_dir = os.path.join(DATA_DIR, args.dataset)

    # Train or evaluate
    if args.command == "train":

        # Load training and validation set
        if args.dataset != "speed":
            dataset_train = urso.Urso()
            dataset_train.load_dataset(dataset_dir, model.config, "train")
            dataset_val = urso.Urso()
            dataset_val.load_dataset(dataset_dir, model.config, "val")
        else:
            pass

        train(model, dataset_train, dataset_val)

    elif args.command == "test":

            dataset = urso.Urso()
            dataset.load_dataset(dataset_dir, config, "test")
            detect_dataset(model, dataset, 100)

    elif args.command == "evaluate":

        dataset_test = urso.Urso()
        dataset_test.load_dataset(dataset_dir, config, "test")

        #evaluate(model, dataset_test)
        evaluate(model, dataset_test, start_index=args.start_index, end_index=args.end_index, img_no=args.img_no)


    else:
        print("wrong command")



############# takes command line argument
#python pose_estimator_icra_two.py evaluate --dataset Datasets_UE_satellite/  
# --weights dataset20240522T1533 --image_scale 0.5 --ori_resolution 24 --ori_param euler_angles --start_index 0 --end_index 12  --img_no batch_size of image (multiple of 4)
