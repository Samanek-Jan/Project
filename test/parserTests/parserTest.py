from unittest import TestCase
import unittest

from data.parser.function_parser import FunctionParser
from data.parser.struct_parser import StructParser
from data.parser.class_parser import ClassParser 

class ParsingTestCase(TestCase):
    
    def test_function_parsing(self):
        function_parser = FunctionParser()
        with open("./test/test_data/cuda_function.cu", "r") as fd:
            lines = fd.readlines()

        parsed_function = function_parser.process(lines)
        self.assertIsNotNone(parsed_function.comment)
        self.assertIsNotNone(parsed_function.header)
        self.assertIsNotNone(parsed_function.code)
        self.assertTrue(parsed_function.is_gpu)
        
    def test_struct_parsing(self):
        struct_parser = StructParser()
        with open("./test/test_data/struct.cpp", "r") as fd:
            lines = fd.readlines()

        parsed_struct = struct_parser.process(lines)
        self.assertIsNotNone(parsed_struct.comment)
        self.assertIsNone(parsed_struct.methods)
        self.assertIsNotNone(parsed_struct.code)
        
    def test_class_parsing(self):
        struct_parser = StructParser()
        with open("./test/test_data/class.cpp", "r") as fd:
            lines = fd.readlines()

        parsed_struct = struct_parser.process(lines)
        self.assertIsNotNone(parsed_struct.comment)
        self.assertIsNotNone(parsed_struct.methods)
        self.assertEqual(len(parsed_struct.methods), 2)
        self.assertIsNotNone(parsed_struct.code)
        

        
if __name__ == "__main__":
    unittest.main()
