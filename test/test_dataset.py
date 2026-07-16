import unittest
from datasets.DAFR_dataset import DAFRDataSet
import cv2
from torch.utils.data import DataLoader
import os
class MyTestCase(unittest.TestCase):
    # def test_somthings(self):
    #     dataset = DAFRDataSet(gt_dir="data/GT", aug_dir="data/dataset")
    #     print(len(dataset))
    def setUp(self):
        self.gt_dir = "data/GT"
        self.aug_dir = "data/dataset"
        self.dataset = DAFRDataSet(gt_dir=self.gt_dir, aug_dir=self.aug_dir)
        
    # def test_somthing1(self):
        
    #     aug_tensor, gt_tensor, aut_src, gt_str = self.dataset[0]
    #     print(aug_tensor.shape)
    #     print(gt_tensor.shape)
    #     print(gt_tensor.min().item())
    #     print(gt_tensor.max().item())
    #     print(aut_src, "<-------->" , gt_str)
    #     aut_image = cv2.imread(aut_src)
    #     gt_image = cv2.imread(gt_str)
    #     cv2.imshow("augimage", aut_image)
    #     cv2.imshow("gtimage", gt_image)
    #     cv2.waitKey()
    #     cv2.destroyAllWindows()
        
        
    def test_something2(self):
        loader = DataLoader(self.dataset, batch_size=4, shuffle=True, num_workers=0)

        batch = next(iter(loader))
        aug_batch, gt_batch, aug_paths, gt_paths = batch

        # print("Aug batch shape:", aug_batch.shape)
        # print("GT batch shape :", gt_batch.shape)
        print(aug_paths, "<--------------->", gt_paths)
        # for i in range(len(aug_paths)):
        #     print(f"{i}:")
        #     print("  AUG:", os.path.basename(aug_paths[i]))
        #     print("  GT :", os.path.basename(gt_paths[i]))
