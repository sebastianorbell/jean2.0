import unittest

class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def test_hello_world(self):
        print("Hello World!")

if __name__ == '__main__':
    unittest.main()
