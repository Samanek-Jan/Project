from unittest import TestCase
import unittest

from data.parser.parser import Parser

from tqdm import tqdm
import os, sys, json

class ParsingTestCase(TestCase):
    def test_parser(self):
        parser = Parser()
        in_str = """
template<typename T>
__device__ inline float toFloat(T value)
{
    return static_cast<float>(value);
}

template <typename T>
__device__ inline int toInt(T value)
{
    return static_cast<int>(value);
}

template <typename T>
__device__ inline uint64_t toUInt64(T value)
{
    return static_cast<uint64_t>(value);
}
        """
        
        parsed_objects, metadata = parser.process_str(in_str, "filename")
        
        self.assertEqual(len(parsed_objects), 3)
        self.assertEqual(parsed_objects[0]["type"], "class")
        self.assertEqual(parsed_objects[1]["type"], "struct")
        self.assertEqual(parsed_objects[2]["type"], "function")

        
if __name__ == "__main__":
    unittest.main()
