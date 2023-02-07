import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
from processing.process import get_image_context_bbox,get_motion_img

class Track2CustomDataset(Dataset):
    def __init__(self, data_tracks, transforms):
        samples = []    
        for key, sample in tqdm(data_tracks.items(), total=len(data_tracks.items())):
            
            frames_paths = sample['frames']
            frames_boxes = sample['boxes']
            samples += [(frames_paths[idx], frames_boxes[idx], frames_paths, frames_boxes) for idx in range(len(frames_paths))]
        self.samples = samples
        self.transforms = transforms
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        frame_path, frame_boxes, all_frames, all_boxes = self.samples[index]

        context, image = get_image_context_bbox(frame_path, frame_boxes)
        motion, motion_line = get_motion_img(all_frames, all_boxes)

        sample = {'image': image.astype(np.float32), 'context': context.astype(np.float32), 
                  'motion': motion.astype(np.float32), 'motion_line': motion_line.astype(np.float32)}

        if self.transforms:
            sample['image'] = self.transforms(sample['image'])
            sample['context'] = self.transforms(sample['context'])
            sample['motion'] = self.transforms(sample['motion'])
            sample['motion_line'] = self.transforms(sample['motion_line'])

        return sample


def get_transforms(img_size, train, size=1):
    if train:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(img_size * size, scale=(0.8, 1)),
            transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    else:
        return transforms.Compose([
            transforms.Resize(img_size * size, img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])