import unittest
from datasets.DAFR_dataset import DAFRDataSet
import cv2

class MyTestCase(unittest.TestCase):
    # def test_somthings(self):
    #     dataset = DAFRDataSet(gt_dir="data/GT", aug_dir="data/dataset")
    #     print(len(dataset))
    
    def test_somthing1(self):
        dataset = DAFRDataSet(gt_dir="data/GT", aug_dir="data/dataset")
        aug_tensor, gt_tensor, aut_src, gt_str = dataset[0]
        print(aug_tensor.shape)
        print(gt_tensor.shape)
        print(gt_tensor.min().item())
        print(gt_tensor.max().item())
        print(aut_src, "<-------->" , gt_str)
        aut_image = cv2.imread(aut_src)
        gt_image = cv2.imread(gt_str)
        cv2.imshow("augimage", aut_image)
        cv2.imshow("gtimage", gt_image)
        cv2.waitKey()
        cv2.destroyAllWindows()
        
        
    def test_something2(self):
        pass