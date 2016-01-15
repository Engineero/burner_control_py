'''
Created on Jan 7, 2016

@author: nathantoner
'''
import unittest
import burner_control.main as bc_main


class TestMain(unittest.TestCase):


  def test_main_returns_true(self):
    result = bc_main.run_simulation()
    self.assertEqual(True, result, "Main program does not return correct result.")


if __name__ == "__main__":
  #import sys;sys.argv = ['', 'Test.test_main']
  unittest.main()