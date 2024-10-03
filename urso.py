'''
Class to handle Unreal datasets
'''
import os
import csv
import numpy as np
import os.path
import skimage
import pandas as pd
import se3lib
import utils
import re
from dataset import Dataset
from se3lib import euler2quat

class Camera:
    fov_x = 90.0 * np.pi / 180
    fov_y = 90.0 * np.pi / 180
    width = 1920  # number of horizontal[pixels]
    height = 1080  # number of vertical[pixels]
    # Focal lengths
    fx = width / (2 * np.tan(fov_x / 2))
    fy = - height / (2 * np.tan(fov_y / 2))

    K = np.matrix([[fx, 0, width / 2], [0, fy, height / 2], [0, 0, 1]])

# Image mean (RGB)
MEAN_PIXEL = np.array([45, 49, 52])

class Urso(Dataset):

    def load_dataset(self, dataset_dir, config, subset):
        """Load a subset of the dataset.
        dataset_dir: The root directory of the dataset.
        subset: What to load (train, val, test)
        """
        self.name = 'Urso'
        if not os.path.exists(dataset_dir):
            print(f"Image directory '{dataset_dir}' not found.")
            return None

        # Load the list of images
        set_filename = os.path.join(dataset_dir, f"{subset}_images.csv")
        rgb_list_df = pd.read_csv(set_filename, names=['filename'], delimiter="\t",  header=None)
        rgb_list = list(rgb_list_df['filename'])

        # Load camera parameters
        self.camera = Camera()

        # Load poses (positions and orientations)
        poses_filename = os.path.join(dataset_dir, f"{subset}_poses_gt.csv")
        
        #poses_df = pd.read_csv(poses_filename)


        data = []

        
        with open(poses_filename, 'r') as file:
            header = file.readline().strip().split(",")  # Extract the header
            
            # Use regex to match both cases of image names with and without commas
            for line in file.readlines():
                match = re.match(r"(Image_.*?\.png),(.*)", line.strip())
                if match:
                    image_file = match.group(1)  # Image file name
                    rest_of_data = match.group(2).split(",")  # The remaining data (x, y, z, etc.)
                    data.append([image_file] + rest_of_data)  # Append to the list

        # Define column names
        columns = ["ImageFile", "x", "y", "z", "wx", "wy", "wz", "t"]

        # Convert to DataFrame
        poses_df = pd.DataFrame(data, columns=columns)

        poses_df[['x', 'y', 'z', 'wx', 'wy', 'wz']] = poses_df[['x', 'y', 'z', 'wx', 'wy', 'wz']].astype(float)


        # Separate columns: filenames, positions (x, y, z), and orientations (wx, wy, wz)
        image_files = poses_df['ImageFile']
        positions = poses_df[['x', 'y', 'z']].to_numpy()
        orientations_euler = poses_df[['wx', 'wy', 'wz']].to_numpy()  # Euler angles

        print(f'Image files : {image_files}')
        # print(f"Positions : {positions}")
        # print(f"Orientation : {orientations_euler}")

        # Initialize arrays for quaternions and translations
        nr_instances = len(image_files)
        q_array = np.zeros((nr_instances, 4), dtype=np.float32)
        t_array = np.zeros((nr_instances, 3), dtype=np.float32)

        print(f'No of images {nr_instances}')

        # Convert Euler angles (wx, wy, wz) to quaternions
        for i, (wx, wy, wz) in enumerate(orientations_euler):
            # Convert Euler angles to quaternions
            q = np.asarray(euler2quat(wx, wy, wz)).flatten()
            #print(q)
            q_array[i, :] = q if q[3] >= 0 else -q  # Ensure quaternion is in the northern hemisphere
            t_array[i, :] = positions[i]  # Store position (x, y, z)

        # Process encoded orientation if not regressing orientation directly
        if not config.REGRESS_ORI:
            ori_encoded, ori_histogram_map, ori_output_mask = utils.encode_ori(
                q_array, config.ORI_BINS_PER_DIM, config.BETA,
                np.array([-180, -90, -180]), np.array([180, 90, 180])
            )
            self.ori_histogram_map = ori_histogram_map
            self.ori_output_mask = ori_output_mask

        # Process encoded location if not regressing location directly
        if not config.REGRESS_LOC:
            img_x_array = positions[:, 1] / positions[:, 0]
            img_y_array = positions[:, 2] / positions[:, 0]
            z_array = positions[:, 0]

            theta_x = self.camera.fov_x * np.pi / 360
            theta_y = self.camera.fov_y * np.pi / 360
            x_max = np.tan(theta_x)
            y_max = np.tan(theta_y)
            z_min = min(z_array)
            z_max = max(z_array)

            loc_encoded, loc_histogram_map = utils.encode_loc(
                np.stack((img_x_array, img_y_array, z_array), axis=1),
                config.LOC_BINS_PER_DIM, config.BETA,
                np.array([-x_max, -y_max, z_min]), np.array([x_max, y_max, z_max])
            )

            self.histogram_3D_map = loc_histogram_map

        if not rgb_list:
            print('No image files found.')
            return None

        # Iterate over the image files and populate the dataset
        for i, file_name in enumerate(image_files):
            q = q_array[i, :]

            # Convert quaternion to angle-axis and Euler angles
            v, theta = se3lib.quat2angleaxis(q)
            pyr = se3lib.quat2euler(q)

            # Prepare the encoded location and orientation data
            ori_encoded_i = [] if config.REGRESS_ORI else ori_encoded[i, :]
            loc_encoded_i = [] if config.REGRESS_LOC else loc_encoded[i, :]

            # Image file path
            rgb_path = os.path.join(dataset_dir, file_name)

            # Add the image information to the dataset
            self.add_image(
                "URSO",
                image_id=i,
                image_name=file_name,
                path=rgb_path,
                location=positions[i],  # (x, y, z)
                quaternion=q,  # Quaternion representing orientation
                angleaxis=[v[0] * theta, v[1] * theta, v[2] * theta],
                pyr=pyr,  # Euler angles (pitch, yaw, roll)
                location_map=loc_encoded_i,
                ori_map=ori_encoded_i
            )

        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.


        """
        print(f'Image id {image_id}')
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image
