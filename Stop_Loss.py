import requests
import Get_Price
import Cancel_Order

def Stop_Loss(symbol,Entry):
  if(Get_Price.get_current_price - Entry)/Entry <= 0.03:
    Cancel_Order.cancel_market_order(symbol)
    
  
  
