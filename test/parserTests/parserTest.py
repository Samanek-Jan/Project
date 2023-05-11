from unittest import TestCase
import unittest

from data.parser.parser import Parser

from tqdm import tqdm
import os, sys, json

class ParsingTestCase(TestCase):
    
    def test_HDS(self):
        with open("test/test_data/templated_func.cu", "r") as fd:
            in_data = fd.read()
        parser = Parser()
        kernels, metadata = parser.process_str(in_data, "filename")
        
        self.assertEqual(len(kernels), 1)

    def test_AS(self):
        with open("test/test_data/invalid_func.cu", "r") as fd:
            in_data = fd.read()
        parser = Parser()
        kernels, metadata = parser.process_str(in_data, "filename")
        
        self.assertEqual(len(kernels), 1)

        
if __name__ == "__main__":
    unittest.main()
