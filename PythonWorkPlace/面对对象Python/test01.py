import unittest
from circle import Circle


class TestCircle(unittest.TestCase):
    def test_get_length(self):
        my_circle = Circle(2)
        length = my_circle.getlength()
        self.assertEqual(length, 12.56)

    def test_get_area(self):
        pass

