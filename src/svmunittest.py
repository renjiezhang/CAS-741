#import unittest
#from StockPredict_test import *


#class TestMathFunc(unittest.TestCase):

    
    #def test_volatility(self):
        priceArray=[100,110,120,130,140,150,160,170,180,190,200]
        result=GetPriceVolatility(1,5,priceArray)
        print(result)
        self.assertEqual(5, len(result))
        
   # def test_momentum(self):
        priceArray=[100,110,120,130,140,150,160,170,180,190,200]
        result=GetMomentum(1,5,priceArray)
        print(result)
        successresult=[1.0, 1.0, 1.0, 1.0, 1.0]
        self.assertEqual(successresult, result)

#if __name__ == '__main__':
    #unittest.main(argv=['first-arg-is-ignored'], exit=False)