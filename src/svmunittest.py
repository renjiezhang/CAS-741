import unittest
from StockPredict_test import *
class TestMathFunc(unittest.TestCase):
    def test_volatility(self):
        price_array=[100,110,120,130,140,150,160,170,180,190,200]
        result=get_price_volatility(1,5,price_array)
        print(result)
        self.assertEqual(5, len(result))
        
    def test_momentum(self):
        price_array=[100,110,120,130,140,150,160,170,180,190,200]
        result=get_momentum(1,5,price_array)
        print(result)
        success_result=[1.0, 1.0, 1.0, 1.0, 1.0]
        self.assertEqual(success_result, result)

    def test_predict(self):
        ndxtdf = data_input('dataset/NDAQ.csv')
        ndxtdf = ndxtdf.sort_values(by='Date', ascending=True)
        ndxt_prices = list(ndxtdf['Close'])
        ndxt_volatility_array =  get_price_volatility(1,5, ndxt_prices)
        ndxt_momentum_array =  get_momentum(1,5, ndxt_prices)
        result=predict('GOOG',1,5,ndxt_volatility_array,ndxt_momentum_array)
        print(result)
        #self.assertEqual(success_result, result)
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)