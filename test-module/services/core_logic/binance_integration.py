# # Na początku pliku, po importach
# import binance
#
#
# from binance.client import Client
# from binance.exceptions import BinanceAPIException, BinanceOrderException
# import os
# import logging
#
# # ============================================================================
# # WALLET METHODS - BINANCE TESTNET
# # ============================================================================
#
# # Globalna instancja klienta Binance (inicjalizowana w __init__ wallet)
# _binance_client = None
# _binance_logger = None
#
#
# def testnet_wallet_init(self):
#     """Inicjalizuje klienta Binance Testnet w wallet."""
#     global _binance_client, _binance_logger
#
#     if _binance_client is None:
#         from configuration.configuration import (
#             BINANCE_TESTNET_API_KEY,
#             BINANCE_TESTNET_API_SECRET
#         )
#
#         if not BINANCE_TESTNET_API_KEY or not BINANCE_TESTNET_API_SECRET:
#             raise ValueError("BINANCE_TESTNET_API_KEY and BINANCE_TESTNET_API_SECRET must be set in .env")
#
#         _binance_client = Client(
#             api_key=BINANCE_TESTNET_API_KEY,
#             api_secret=BINANCE_TESTNET_API_SECRET,
#             testnet=True  # Używa https://testnet.binance.vision
#         )
#
#         # Logger
#         base_dir = os.getcwd()
#         log_dir = os.path.join(base_dir, "shared", "logs", "binance_trading")
#         os.makedirs(log_dir, exist_ok=True)
#
#         from datetime import datetime
#         log_file = os.path.join(
#             log_dir,
#             f"testnet_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
#         )
#
#         _binance_logger = logging.getLogger("BinanceTestnet")
#         _binance_logger.setLevel(logging.INFO)
#
#         file_handler = logging.FileHandler(log_file)
#         file_handler.setFormatter(
#             logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
#         )
#         console_handler = logging.StreamHandler()
#         console_handler.setFormatter(
#             logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
#         )
#
#         _binance_logger.addHandler(file_handler)
#         _binance_logger.addHandler(console_handler)
#         _binance_logger.info("✅ Binance Testnet client initialized")
#
#     # Zapisz referencję w wallet
#     self.binance_client = _binance_client
#     self.binance_logger = _binance_logger
#
#
# def testnet_wallet_get_symbol_info(self, symbol: str):
#     """Pobiera informacje o symbolu (min qty, step size, etc.)."""
#     try:
#         exchange_info = self.binance_client.get_exchange_info()
#         for s in exchange_info.get("symbols", []):
#             if s["symbol"] == symbol.upper():
#                 return s
#         return None
#     except Exception as e:
#         self.binance_logger.error(f"❌ Failed to get symbol info: {e}")
#         return None
#
#
# def testnet_wallet_format_quantity(self, symbol: str, quantity: float) -> float:
#     """Formatuje ilość zgodnie z wymaganiami symbolu (step size)."""
#     symbol_info = self.testnet_wallet_get_symbol_info(symbol)
#     if not symbol_info:
#         return quantity
#
#     for f in symbol_info.get("filters", []):
#         if f.get("filterType") == "LOT_SIZE":
#             step_size = float(f.get("stepSize", "0.001"))
#             return float(int(quantity / step_size) * step_size)
#
#     return quantity
#
#
# def testnet_wallet_buy(self, order):
#     """
#     Wykonuje zlecenie BUY na Binance Testnet.
#     Zastępuje standardową metodę buy() w wallet.
#     """
#     global _binance_client, _binance_logger
#
#     # Inicjalizuj klienta jeśli potrzeba
#     if not hasattr(self, 'binance_client') or self.binance_client is None:
#         testnet_wallet_init(self)
#
#     try:
#         symbol = order.symbol.upper()
#         qty = self.testnet_wallet_format_quantity(symbol, order.qty)
#
#         if qty <= 0:
#             self.binance_logger.error(f"❌ Invalid quantity: {qty}")
#             order.rejected("invalid_quantity")
#             return
#
#         # Sprawdź balans USDT
#         balance_info = self.binance_client.get_asset_balance(asset="USDT")
#         balance = float(balance_info.get("free", 0.0)) if balance_info else 0.0
#
#         required = order.price * qty * 1.001  # + fee
#
#         if required > balance:
#             self.binance_logger.error(f"❌ Insufficient balance: {balance} < {required}")
#             order.rejected("insufficient_balance")
#             return
#
#         # Wykonaj zlecenie MARKET
#         result = self.binance_client.create_order(
#             symbol=symbol,
#             side=Client.SIDE_BUY,
#             type=Client.ORDER_TYPE_MARKET,
#             quantity=qty
#         )
#
#         # Pobierz rzeczywistą cenę wykonania z fills
#         fills = result.get("fills", [])
#         if fills:
#             executed_price = float(fills[0].get("price", order.price))
#             executed_qty = float(fills[0].get("qty", qty))
#         else:
#             executed_price = order.price
#             executed_qty = qty
#
#         self.binance_logger.info(
#             f"✅ BUY executed: {symbol} qty={executed_qty} price={executed_price} "
#             f"orderId={result.get('orderId')}"
#         )
#
#         # Aktualizuj stan wallet (jak w oryginalnej metodzie)
#         order.slippage_price = executed_price
#         current_qty = self.holdings.get(symbol, 0.0)
#         current_avg = self.avg_buy_price.get(symbol, 0.0)
#
#         if current_qty > 0:
#             new_avg = ((current_qty * current_avg) + (executed_qty * executed_price)) / (current_qty + executed_qty)
#             self.avg_buy_price[symbol] = new_avg
#         else:
#             self.avg_buy_price[symbol] = executed_price
#
#         self.holdings[symbol] = self.holdings.get(symbol, 0.0) + executed_qty
#
#         # Aktualizuj balance (pobierz z Binance)
#         balance_info = self.binance_client.get_asset_balance(asset="USDT")
#         self.balance = float(balance_info.get("free", 0.0)) if balance_info else 0.0
#
#         self.buys_count += 1
#         order.status = "accepted"
#
#     except BinanceOrderException as e:
#         self.binance_logger.error(f"❌ Binance order error: {e}")
#         order.rejected(f"binance_error: {str(e)}")
#     except BinanceAPIException as e:
#         self.binance_logger.error(f"❌ Binance API error: {e}")
#         order.rejected(f"api_error: {str(e)}")
#     except Exception as e:
#         self.binance_logger.error(f"❌ Unexpected error: {e}", exc_info=True)
#         order.rejected(f"unexpected_error: {str(e)}")
#
#
# def testnet_wallet_sell(self, order):
#     """
#     Wykonuje zlecenie SELL na Binance Testnet.
#     Zastępuje standardową metodę sell() w wallet.
#     """
#     global _binance_client, _binance_logger
#
#     # Inicjalizuj klienta jeśli potrzeba
#     if not hasattr(self, 'binance_client') or self.binance_client is None:
#         testnet_wallet_init(self)
#
#     try:
#         symbol = order.symbol.upper()
#         qty = self.testnet_wallet_format_quantity(symbol, order.qty)
#
#         if qty <= 0:
#             self.binance_logger.error(f"❌ Invalid quantity: {qty}")
#             order.rejected("invalid_quantity")
#             return
#
#         # Sprawdź holdings (pobierz z Binance)
#         base_asset = symbol.replace("USDT", "")
#         balance_info = self.binance_client.get_asset_balance(asset=base_asset)
#         balance = float(balance_info.get("free", 0.0)) if balance_info else 0.0
#
#         if qty > balance:
#             self.binance_logger.error(f"❌ Insufficient holdings: {balance} < {qty}")
#             order.rejected("insufficient_holdings")
#             return
#
#         # Wykonaj zlecenie MARKET
#         result = self.binance_client.create_order(
#             symbol=symbol,
#             side=Client.SIDE_SELL,
#             type=Client.ORDER_TYPE_MARKET,
#             quantity=qty
#         )
#
#         # Pobierz rzeczywistą cenę wykonania
#         fills = result.get("fills", [])
#         if fills:
#             executed_price = float(fills[0].get("price", order.price))
#             executed_qty = float(fills[0].get("qty", qty))
#         else:
#             executed_price = order.price
#             executed_qty = qty
#
#         self.binance_logger.info(
#             f"✅ SELL executed: {symbol} qty={executed_qty} price={executed_price} "
#             f"orderId={result.get('orderId')}"
#         )
#
#         # Aktualizuj stan wallet
#         order.slippage_price = executed_price
#         self.holdings[symbol] = self.holdings.get(symbol, 0.0) - executed_qty
#
#         if self.holdings[symbol] <= 1e-9:
#             self.holdings[symbol] = 0.0
#             self.avg_buy_price[symbol] = 0.0
#
#         # Aktualizuj balance (pobierz z Binance)
#         balance_info = self.binance_client.get_asset_balance(asset="USDT")
#         self.balance = float(balance_info.get("free", 0.0)) if balance_info else 0.0
#
#         self.sells_count += 1
#         order.status = "accepted"
#
#     except BinanceOrderException as e:
#         self.binance_logger.error(f"❌ Binance order error: {e}")
#         order.rejected(f"binance_error: {str(e)}")
#     except BinanceAPIException as e:
#         self.binance_logger.error(f"❌ Binance API error: {e}")
#         order.rejected(f"api_error: {str(e)}")
#     except Exception as e:
#         self.binance_logger.error(f"❌ Unexpected error: {e}", exc_info=True)
#         order.rejected(f"unexpected_error: {str(e)}")
#
#
