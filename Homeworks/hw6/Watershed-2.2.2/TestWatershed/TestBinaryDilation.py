import Watershed
import os
import unittest

class TestBinaryDilation(unittest.TestCase):

    def setUp(self):
        self.shed = Watershed.Watershed(data_image="line.jpg",binary_or_gray_or_color="binary")
        self.shed.extract_data_pixels()

    def test_binary_dilations(self):
        available = False
        print("testing the operation of binary dilations")
        dilated_image = self.shed._dilate_for_unittest(10)
        if os.path.exists("_dilation.jpg"):
            available = True
        self.assertEqual(available, True)
        width,height = dilated_image.size
        rowlist = []
        for i in range(width):
            rowlist.append(dilated_image.getpixel((i,32)))
        self.assertEqual(rowlist[10], 255)
        os.remove("_dilation.jpg")

def getTestSuites(type):
    return unittest.TestSuite([
            unittest.makeSuite(TestBinaryDilation, type)
                             ])                    
if __name__ == '__main__':
    unittest.main()

