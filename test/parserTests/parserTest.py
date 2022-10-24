from unittest import TestCase
import unittest

from data.parser.parser import DATA_FILE_SUFFIX, Parser
from data.parser.function_parser import FunctionParser
from data.parser.parsing_object import PARSING_TYPES
from data.parser.struct_parser import StructParser
from data.parser.class_parser import ClassParser

from tqdm import tqdm
import os, sys, json

class ParsingTestCase(TestCase):
    
    def test_function_parsing(self):
        function_parser = FunctionParser()
        with open("./test/test_data/cuda_function.cu", "r") as fd:
            lines = fd.readlines()

        parsed_function = function_parser.process(lines)
        self.assertGreater(len(parsed_function["comment"]), 0)
        self.assertGreater(len(parsed_function["header"]), 0)
        self.assertGreater(len(parsed_function["body"]), 0)
        self.assertTrue(parsed_function["is_gpu"])
        
    def test_struct_parsing(self):
        struct_parser = StructParser()
        with open("./test/test_data/struct.cpp", "r") as fd:
            lines = fd.readlines()

        parsed_struct = struct_parser.process(lines)
        self.assertGreater(len(parsed_struct["comment"]), 0)
        self.assertGreater(len(parsed_struct["body"]), 0)
        
    def test_class_parsing(self):
        class_parser = ClassParser()
        with open("./test/test_data/class.cpp", "r") as fd:
            lines = fd.readlines()

        parsed_class = class_parser.process(lines)
        self.assertGreater(len(parsed_class["comment"]), 0)
        self.assertGreater(len(parsed_class["body"]), 0)
        
    def test_parser(self):
        parser = Parser()
        filename = "./test/test_data/parser_data.cpp"
        parsed_objects = parser.process_file(filename)
        
        self.assertEqual(len(parsed_objects), 3)
        self.assertEqual(parsed_objects[0]["type"], "class")
        self.assertEqual(parsed_objects[1]["type"], "struct")
        self.assertEqual(parsed_objects[2]["type"], "function")
        
    def test_parse_real_data(self):
        parser = Parser()
        filename = "data/raw/kspaceFirstOrder-CUDA/Containers/CudaMatrixContainer.cu"
        
        parsed_objects = parser.process_file(filename)
        self.assertIsNotNone(parsed_objects)
        self.assertEqual(len(parsed_objects), 7)
        
    def test_check_converted_objects(self):
        in_folder = "data/processed"
        files = [file for file in os.listdir(in_folder) if file.endswith(DATA_FILE_SUFFIX)]
        
        for file in files:
            
            with open(os.path.join(in_folder, file), "r") as fd:
                parsed_objects = json.load(fd)
                
            for parsed_object in parsed_objects:
                self.assertIsNotNone(parsed_object["type"], "Error: type missing in {}".format(file))
                self.assertIsNotNone(parsed_object["body"], "Error: body missing in {}".format(file))

                self.assertIn(parsed_object["type"], PARSING_TYPES, "Error: Unknown type \"{}\" in file \"{}\"".format(parsed_object["type"], file))

                if parsed_object["type"] == "function":
                    self.assertGreater(len(parsed_object["header"]), 0, "Error: header empty in file \"{}\"".format(file))                
                    self.assertGreater(len(parsed_object["body"]), 0, "Error: body empty in file \"{}\"".format(file))                
        

        
if __name__ == "__main__":
    unittest.main()
