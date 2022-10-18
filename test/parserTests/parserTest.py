from unittest import TestCase
import unittest

from data.parser.parser import Parser
from data.parser.function_parser import FunctionParser
from data.parser.parsing_object import ParsedClass, ParsedFunction, ParsedStruct
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
        self.assertIsNotNone(parsed_struct.code)
        
    def test_class_parsing(self):
        class_parser = ClassParser()
        with open("./test/test_data/class.cpp", "r") as fd:
            lines = fd.readlines()

        parsed_class = class_parser.process(lines)
        self.assertIsNotNone(parsed_class.comment)
        self.assertIsNotNone(parsed_class.code);
        
    def test_parser(self):
        parser = Parser()
        filename = "./test/test_data/parser_data.cpp"
        parsed_objects = parser.process_file(filename)
        
        self.assertEqual(len(parsed_objects), 3)
        self.assertEqual(parsed_objects[0].__class__, ParsedClass)
        self.assertEqual(parsed_objects[1].__class__, ParsedStruct)
        self.assertEqual(parsed_objects[2].__class__, ParsedFunction)
        
    def test_parse_real_data(self):
        parser = Parser()
        filename = "data/raw/kspaceFirstOrder-CUDA/Containers/CudaMatrixContainer.cu"
        
        parsed_objects = parser.process_file(filename)
        self.assertIsNotNone(parsed_objects)
        self.assertEqual(len(parsed_objects), 5)
        

        
if __name__ == "__main__":
    unittest.main()
