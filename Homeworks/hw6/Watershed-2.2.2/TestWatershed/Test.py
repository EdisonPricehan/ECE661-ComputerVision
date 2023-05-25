#!/usr/bin/env python

import unittest
import TestImageLoadingAndDataExtraction
import TestBinaryDilation

class WatershedTestCase( unittest.TestCase ):
    def checkVersion(self):
        import Watershed

testSuites = [unittest.makeSuite(WatershedTestCase, 'test')] 

for test_type in [
            TestImageLoadingAndDataExtraction,
            TestBinaryDilation,
    ]:
    testSuites.append(test_type.getTestSuites('test'))


def getTestDirectory():
    try:
        return os.path.abspath(os.path.dirname(__file__))
    except:
        return '.'

import os
os.chdir(getTestDirectory())

runner = unittest.TextTestRunner()
runner.run(unittest.TestSuite(testSuites))
