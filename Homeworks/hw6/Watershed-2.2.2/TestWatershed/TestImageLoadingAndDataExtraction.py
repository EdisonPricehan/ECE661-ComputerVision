import Watershed
import unittest

class TestImageLoadingAndDataExtraction(unittest.TestCase):

    def setUp(self):
        self.shed = Watershed.Watershed(data_image="line.jpg",binary_or_gray_or_color="binary")
        self.shed.extract_data_pixels()

    def test_loaded_image(self):
        print("testing image loading and pixel data extraction")
        self.shed.displayImage(self.shed.data_im)
        width,height = self.shed.data_im.size
        val = self.shed.data_im.getpixel((30,15))
        self.assertEqual(width, 40)
        self.assertEqual(height, 40)
        self.assertEqual(val,255)

def getTestSuites(type):
    return unittest.TestSuite([
            unittest.makeSuite(TestImageLoadingAndDataExtraction, type)
                             ])                    
if __name__ == '__main__':
    unittest.main()

