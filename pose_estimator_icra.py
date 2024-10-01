import os
import numpy as np
import skimage
import pandas as pd
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import se3lib
import torch

import utils
import net
from config import Config
import urso
from tqdm import tqdm

# Ensure that your model and tensors are on the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device for pose estimation: {device}")

# Configuration and setup (replaces command-line arguments)
command = "test"  
dataset = "Datasets_UE_satellite"  # Name of the dataset
epochs = 100  
image_scale = 0.5  # Image scaling factor
ori_param = "euler_angles"  # Orientation parameter: 'quaternion', 'euler_angles', 'angle_axis'
ori_resolution = 24  # Orientation resolution
weights = "dataset20240522T1533/weights_dataset_0100"  # Path to the trained model weights
bottleneck_width = 32
batch_size = 1
learning_rate = 0.001
loss_weights = {"loc_loss": 1.0, "ori_loss": 1.0}
image_batch_size = 10

# Initalize Directories
MODEL_DIR = os.path.abspath("./models")
DEFAULT_LOGS_DIR = os.path.join(MODEL_DIR, "logs")
save_dir = "datasets/Graphs/"
DATA_DIR = os.path.abspath("UrsoNet-master/datasets/")
COCO_WEIGHTS_PATH = os.path.join(MODEL_DIR, "mask_rcnn_coco.h5")


# Initialize config
config = Config()
config.ORIENTATION_PARAM = ori_param
config.ORI_BINS_PER_DIM = ori_resolution
config.NAME = dataset
config.EPOCHS = epochs
config.BOTTLENECK_WIDTH = bottleneck_width
config.LEARNING_RATE = learning_rate
config.NR_DENSE_LAYERS = 1
config.IMAGE_RESIZE_MODE = 'pad64'
config.BATCH_SIZE  = 1



# Image processing setup
width_original = urso.Camera.width
height_original = urso.Camera.height

config.IMAGE_MAX_DIM = round(width_original * image_scale)
config.IMAGE_MIN_DIM = round(height_original * image_scale)

if config.IMAGE_MAX_DIM % 64 > 0:
    raise Exception("Scale problem. Image maximum dimension must be dividable by 2 at least 6 times.")

# n.b: assumes height is less than width
height_scaled = round(height_original * image_scale)
if height_scaled % 64 > 0:
    config.IMAGE_MIN_DIM = height_scaled - height_scaled%64 + 64
else:
    config.IMAGE_MIN_DIM = height_scaled


if command == "train":
        config.IMAGES_PER_GPU = batch_size
else:
    config.IMAGES_PER_GPU = 1

config.BATCH_SIZE = config.IMAGES_PER_GPU * 1

config.update()

# Load the dataset and the trained model
dataset_dir = os.path.join(DATA_DIR, dataset)
dataset_test = urso.Urso()
dataset_test.load_dataset(dataset_dir, config, "test")

# Check if dataset_test loaded images properly
print(f"Loaded {len(dataset_test.image_ids)} images for testing.")

# Define the range of indices for inference
start_idx = 50  # Start at this index (e.g., 50th image)
end_idx = 60  # Stop at this index (e.g., 60th image)

# Ensure that end_idx is not beyond the number of images
end_idx = min(end_idx, len(dataset_test.image_ids))


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

def test_and_submit(model, dataset_virtual, dataset_real):
    """ Evaluates model on ESA challenge test-set (no labels)
    and outputs submission file in a format compatible with the ESA server (probably down by now)
    """

    # ESA API
    from submission import SubmissionWriter
    submission = SubmissionWriter()

    # TODO: Make the next 2 loops a nested loop

    # Synthetic test set
    for image_id in dataset_virtual.image_ids:

        print('Image ID:', image_id)

        image = dataset_virtual.load_image(image_id)
        info = dataset_virtual.image_info[image_id]

        results = model.detect([image], verbose=1)

        # Retrieve location
        if model.config.REGRESS_LOC:
            loc_est = results[0]['loc']
        else:
            loc_pmf = utils.stable_softmax(results[0]['loc'])

            # Compute location mean according to first moment
            loc_est = np.asmatrix(loc_pmf) * np.asmatrix(dataset_virtual.histogram_3D_map)

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
            q_est, q_est_cov = se3lib.quat_weighted_avg(dataset_virtual.ori_histogram_map, ori_pmf)

        # Change quaternion order
        q_rect = [q_est[3], q_est[0], q_est[1], q_est[2]]

        submission.append_test(info['path'].split('/')[-1], q_rect, loc_est)

    # Real test set

    for image_id in dataset_real.image_ids:

        print('Image ID:', image_id)

        image = dataset_real.load_image(image_id)
        info = dataset_real.image_info[image_id]

        results = model.detect([image], verbose=1)

        # Retrieve location
        if model.config.REGRESS_LOC:
            loc_est = results[0]['loc']
        else:
            loc_pmf = utils.stable_softmax(results[0]['loc'])

            # Compute location mean according to first moment
            loc_est = np.asmatrix(loc_pmf) * np.asmatrix(dataset_real.histogram_3D_map)

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
            q_est, q_est_cov = se3lib.quat_weighted_avg(dataset_real.ori_histogram_map, ori_pmf)

        # Change quaternion order
        q_rect = [q_est[3], q_est[0], q_est[1], q_est[2]]

        submission.append_real_test(info['path'].split('/')[-1], q_rect, loc_est)

    submission.export(suffix='debug')
    print('Submission exported.')


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

        # print('Loc Error: ', loc_err)
        # print('Ori Error: ', angular_err)

        # Compute ESA score
        esa_score = loc_err/np.linalg.norm(loc_gt) + 2*np.arccos(np.abs(np.asmatrix(q_est)*np.asmatrix(q_gt).transpose()))
        esa_scores_acc.append(esa_score)

        # Store depth
        distances_acc.append(loc_gt[2])

    # print('Mean est. location error: ', np.mean(loc_err_acc))
    # print('Mean est. orientation error: ', np.mean(ori_err_acc))
    # print('ESA score: ', np.mean(esa_scores_acc))
    # print('Mean encoded location error: ', np.mean(loc_encoded_err_acc))

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
            nr_EM_iterations = 5
            Q_mean, Q_var, Q_priors, model_scores = fit_GMM_to_orientation(dataset.ori_histogram_map, ori_pmf, nr_EM_iterations, var)
            print('Multimodal errors',2 * np.arccos(np.abs(np.asmatrix(Q_mean) * np.asmatrix(q_gt).transpose())) * 180 / np.pi)
            
            q_est_1 = Q_mean[0, :]
            q_est_2 = Q_mean[1, :]
            utils.polar_plot(q_est_1, q_est_2)

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
        if not model.config.REGRESS_ORI:

            nr_bins_per_dim = model.config.ORI_BINS_PER_DIM
            utils.visualize_weights(ori_encoded_gt,ori_pmf,nr_bins_per_dim)

        # Show image
        fig, (ax_1, ax_2) = plt.subplots(1, 2, figsize=(12, 8))
        ax_1.imshow(image_ori)
        ax_1.set_xticks([])
        ax_1.set_yticks([])
        ax_2.imshow(image_ori)
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

        utils.visualize_axes(ax_1, q_gt, loc_gt, K, 100)
        utils.visualize_axes(ax_2, q_est, loc_est, K, 100)

        # Save the figure
        fig.savefig(os.path.join(save_dir, f'detection_{image_id}_axes.png'))
        plt.close(fig)

        #utils.polar_plot(q_gt, q_est)
        plt.savefig(os.path.join(save_dir, f'detection_{image_id}_polar.png'))
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
        fig.savefig(os.path.join(save_dir, f'detection_{image_id}_locations.png'))
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


# Initialize the model for inference
model = net.UrsoNet(mode="inference", config=config, model_dir=MODEL_DIR)

# Load model weights
weights_path = os.path.join(DEFAULT_LOGS_DIR, weights + ".h5")
model.load_weights(weights_path, weights_path, by_name=True)

# Paths to CSV files
test_images_path = os.path.join(dataset_dir, "test_images.csv")
test_pose_gt_path = os.path.join(dataset_dir, "test_poses_gt.csv")

# Load ground truth data
test_images_df = pd.read_csv(test_images_path, delimiter="\t", header=None, names=['ImageFile'])

print(test_images_df.head())


# load test_poses_gt.csv
import re


data = []

with open(test_pose_gt_path, 'r') as file:
    header = file.readline().strip().split(",") 
    

    for line in file.readlines():
        match = re.match(r"(Image_.*?\.png),(.*)", line.strip())
        if match:
            image_file = match.group(1)  # Image file name
            rest_of_data = match.group(2).split(",")  # The remaining data (x, y, z, etc.)
            data.append([image_file] + rest_of_data)  # Append to the list


columns = ["ImageFile", "x", "y", "z", "wx", "wy", "wz", "t"]

test_pose_gt_df = pd.DataFrame(data, columns=columns)

# Specify output file path
output_txt_file = os.path.join(save_dir, f"pose_estimation_results_icra.txt")


# Batch Processing of images 
def evaluate_modified(model, dataset, test_pose_gt_df, start_idx, end_idx, output_txt_file):
    """Evaluate the model and save the loc_est with the least error for each unique loc_gt across multiple images and timesteps."""
    
    write_header = not os.path.exists(output_txt_file)
    
    best_results = {}

    all_image_results = []

    for i, image_id in enumerate(dataset.image_ids[start_idx:end_idx]):
        print(f"Processing image {i+1} (Image ID: {image_id})")

        # Load image details
        image_info = dataset.image_info[image_id]
        image_name = image_info['name']
        image_ori = dataset.load_image(image_id)

        loc_gt = tuple(dataset.load_location(image_id))
        q_gt = dataset.load_quaternion(image_id)

        q_gt_euler = test_pose_gt_df.loc[test_pose_gt_df['ImageFile'] == image_name, ['wx', 'wy', 'wz']].values[0]
        timestep = test_pose_gt_df.loc[test_pose_gt_df['ImageFile'] == image_name, 't'].values[0]

        # Run the model detection
        results = model.detect([image_ori], verbose=1)

        loc_est = results[0]['loc']
        if model.config.REGRESS_ORI:
            q_est = results[0]['ori']
        else:
            ori_pmf = utils.stable_softmax(results[0]['ori'])
            q_est, _ = se3lib.quat_weighted_avg(dataset.ori_histogram_map, ori_pmf)

        #Retrieve orientation
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
            nr_EM_iterations = 5
            Q_mean, Q_var, Q_priors, model_scores = fit_GMM_to_orientation(dataset.ori_histogram_map, ori_pmf, nr_EM_iterations, var)
            print('Multimodal errors',2 * np.arccos(np.abs(np.asmatrix(Q_mean) * np.asmatrix(q_gt).transpose())) * 180 / np.pi)
            
            q_est_1 = Q_mean[0, :]
            q_est_2 = Q_mean[1, :]
            #utils.polar_plot(q_est_1, q_est_2)

        # Compute Errors
        angular_err = 2 * np.arccos(np.abs(np.asmatrix(q_est) * np.asmatrix(q_gt).transpose())) * 180 / np.pi
        loc_err = np.linalg.norm(loc_est - loc_gt)

        q_est = se3lib.quat2euler(q_est)


        # Store the result for this image
        all_image_results.append({
            'image_name': image_name,
            'timestep': timestep,
            'loc_gt': loc_gt,
            'q_gt_euler': q_gt_euler,
            'loc_est': loc_est,
            'q_est': q_est,
            'loc_err': loc_err,
            'angular_err': angular_err
        })

        if loc_gt not in best_results:
            best_results[loc_gt] = {
                'best_loc_est': loc_est,
                'best_loc_err': loc_err
            }
        else:
            # If a smaller location error is found, update the "best" result
            if loc_err < best_results[loc_gt]['best_loc_err']:
                best_results[loc_gt] = {
                    'best_loc_est': loc_est,
                    'best_loc_err': loc_err
                }

    with open(output_txt_file, 'a') as file:
        if write_header:
            file.write("Image ID,timestep,x,y,z,wx,wy,wz,ox,oy,oz,est x,est y, est z, est ox, est oy, est oz,Loc Err,Angular Err\n")

        for result in all_image_results:
            loc_gt = result['loc_gt']
            best_loc_est = best_results[loc_gt]['best_loc_est']  # Get the best loc_est for this loc_gt
            best_loc_err = best_results[loc_gt]['best_loc_err']

            # Prepare strings for writing
            loc_gt_str = ', '.join(map(str, result['loc_gt']))
            q_gt_str = ', '.join(map(str, result['q_gt_euler']))
            loc_est_str = ', '.join(map(str, result['loc_est']))  # Current loc_est
            best_loc_est_str = ', '.join(map(str, best_loc_est))  # Best loc_est 
            q_est_str = ', '.join(map(str, result['q_est']))

            # Write the results for each image
            file.write(f"{result['image_name']}, {result['timestep']}, {loc_gt_str}, {q_gt_str}, {best_loc_est_str}, {q_est_str}, {best_loc_err}, {result['angular_err']}\n")



# Define start and end indices for images to be processed
start_idx = 0
end_idx = 12

evaluate_modified(model, dataset_test, test_pose_gt_df, start_idx, end_idx, output_txt_file)


print(f"Pose estimation completed. Results saved in {output_txt_file}.")