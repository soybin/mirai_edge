# system
import os
import os.path
import sys
# time
import time
import random
from datetime import datetime as date
from datetime import timedelta
# math
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# ai
import tensorflow as tf
from sklearn.utils import class_weight
# finance
import yfinance as yf
import yahoo_fin.stock_info as si

class mirai_edge:
    def __init__(self):
        os.system('cls')
        print("--------------")
        print("MIRAI EDGE 1.2")
        print("--------------")

    def get_rsi(self, data, start_index, end_index, period):
        rsi = []
        for i in range(start_index, end_index + 1):
            avg_up=0
            avg_down=0
            for j in range(i - period, i + 1):
                change_j = data['Close'][j] - data['Close'][j - 1]
                if change_j > 0:
                    avg_up += change_j / period
                else:
                    avg_down += -change_j / period
            rsi_value = 0
            if avg_down == 0:
                rsi_value = 100
            else:
                rs = avg_up / avg_down
                rsi_value = 100 - 100 / (1 + rs)
            rsi.append(rsi_value)
        return rsi

    def get_bbs(self, data, start_index, end_index, period, standard_deviations):
        # ----
        # define mean price per time unit
        # used to compute sma
        # ----
        mean_price = []
        for i in range(0, end_index + 1):
            mean_price.append((data['High'][i] + data['Low'][i] + data['Close'][i]) / 3.0)
        bb_upper = []
        bb_lower = []
        for i in range(start_index, end_index + 1):
            # ----
            # first compute sma value for given day
            # ----
            sma_value = 0.0
            for j in range(i - (period - 1), i + 1):
                sma_value += mean_price[j]
            sma_value = sma_value / period
            # ----
            # then compute bollinger bands
            # ----
            bb_value = 0
            for j in range(0, period):
                bb_value += (mean_price[i - j] - sma_value)**2
            bb_value = math.sqrt(bb_value / period)
            bb_upper.append(sma_value + standard_deviations * bb_value)
            bb_lower.append(sma_value - standard_deviations * bb_value)
        return bb_lower, bb_upper

    def mirai_no_data(self, ticker_data, start_date, end_date, STD_DEVIATIONS, SMA_PERIOD, RSI_PERIOD, RSI_LOW, RSI_HIGH, MAX_HOLD):
        start_index = max(SMA_PERIOD - 1, RSI_PERIOD + 1 - 1) # [+1 because rsi uses previous value as well. -1 because of vector]
        end_index = ticker_data['Close'].size - 1
        # ----
        # get bollinger bands
        # ----
        bb_lower, bb_upper = self.get_bbs(
                data=ticker_data,
                start_index=start_index,
                end_index=end_index,
                period=SMA_PERIOD,
                standard_deviations=STD_DEVIATIONS
        )
        # ----
        # get rsi
        # ----
        rsi = self.get_rsi(
                data=ticker_data,
                start_index=start_index,
                end_index=end_index,
                period=RSI_PERIOD
        )
        # ----
        # get prices
        # ----
        prices_mean = []
        prices_low = []
        prices_high = []
        for i in range(start_index, end_index + 1):
            prices_mean.append((ticker_data['Close'][i] + ticker_data['High'][i] + ticker_data['Low'][i]) / 3.0)
            prices_low.append(ticker_data['Low'][i])
            prices_high.append(ticker_data['High'][i])
        return bb_lower, bb_upper, rsi, prices_mean, prices_low, prices_high
            
    def mirai_no_yaiba(self, x, y, c, MIN_OPERATIONS):
        DETAIL_SLOPE = 20
        DETAIL_INTERCEPT = 20
        MAX_Y_FACTOR = 5.0
        MIN_PERCENTAGE = 0.85
        TRUE_POSITIVE = 1
        TRUE_NEGATIVE = 1
        FALSE_POSITIVE = 0
        FALSE_NEGATIVE = -2
        n = len(x)
        max_y = -1e10
        min_y = 1e10
        for i in range(0, n):
            max_y = max(max_y, y[i])
            min_y = min(min_y, y[i])
        def compute_score(m, b):
            operations = 0
            successes = 0
            score = 0
            for i in range(0, n):
                line_evaluated = m * x[i] + b
                if y[i] >= line_evaluated and c[i] == 1:
                    score = score + TRUE_POSITIVE
                    operations = operations + 1
                    successes = successes + 1
                elif y[i] < line_evaluated and c[i] == 1:
                    score = score + FALSE_POSITIVE
                elif y[i] >= line_evaluated and c[i] == 0:
                    score = score + FALSE_NEGATIVE
                    operations = operations + 1
                elif y[i] < line_evaluated and c[i] == 0:
                    score = score + TRUE_NEGATIVE
            if operations < MIN_OPERATIONS or (successes / operations) < MIN_PERCENTAGE:
                score = -1e11
            return score
        m = 0.0
        b = 0.0
        best_score = 0
        for angle in np.linspace(0, math.pi / 2.0, DETAIL_SLOPE):
            for b_tmp in np.linspace(min_y, max_y * MAX_Y_FACTOR, DETAIL_INTERCEPT):
                # tangent is infinite
                if angle == math.pi / 2.0:
                    break
                m_tmp = -math.tan(angle)
                score = compute_score(m_tmp, b_tmp)
                if score > best_score:
                    m = m_tmp
                    b = b_tmp
                    best_score = score
        return m, b, best_score

    def mirai_no_sentaku(self, start_date, end_date):
        batch_number = 1
        number_of_batches = 1
        if len(sys.argv) > 1:
            batch_number = int(sys.argv[1])
            number_of_batches = int(sys.argv[2])
        # read ticker list
        f = open("./data/tickers.txt", "r")
        tickers = []
        for ticker in f:
            tickers.append(ticker.strip())
        f.close()
        tickers_per_batch = len(tickers) // number_of_batches
        # compute hyperplane per ticker
        for i in range(tickers_per_batch * (batch_number - 1), tickers_per_batch * batch_number):
            ticker = tickers[i]
            print("WORKING ON " + ticker)
            # check if data folder exists
            if not os.path.isdir("./data/" + ticker):
                os.mkdir("./data/" + ticker)
            # EXPECTED MOVE
            EXPECTED_MOVE_MIN = 0.03
            EXPECTED_MOVE_MAX = 0.05
            EXPECTED_MOVE_ITERATIONS = 3
            # LOSS FACTOR
            LOSS_FACTOR_MIN = 2
            LOSS_FACTOR_MAX = 2
            LOSS_FACTOR_ITERATIONS = 1
            # STANDARD DEVIATIONS
            STD_DEVIATIONS_MIN = 1.9
            STD_DEVIATIONS_MAX = 2.1
            STD_DEVIATIONS_ITERATIONS = 3
            # SMA PERIOD
            SMA_PERIOD_MIN = 18
            SMA_PERIOD_MAX = 22
            SMA_PERIOD_STEP = 2
            # RSI PERIOD
            RSI_PERIOD_MIN = 12
            RSI_PERIOD_MAX = 16
            RSI_PERIOD_STEP = 2
            # RSI LOW
            RSI_LOW_MIN = 20
            RSI_LOW_MAX = 40
            RSI_LOW_STEP = 10
            # RSI MAX
            RSI_HIGH_MIN = 60
            RSI_HIGH_MAX = 80
            RSI_HIGH_STEP = 10
            # MAX HOLD
            MAX_HOLD_MIN = 30
            MAX_HOLD_MAX = 30
            MAX_HOLD_STEP = 10
            # OPERATIONS
            MIN_OPERATIONS_MIN_BUY = 10
            MIN_OPERATIONS_MAX_BUY = 10
            MIN_OPERATIONS_MIN_SELL = 10
            MIN_OPERATIONS_MAX_SELL = 10
            MIN_OPERATIONS_STEP = 10
            # save used info
            slope_final_buy= 0
            intercept_final_buy = 0
            best_score_buy = 0
            expected_move_buy = 0
            loss_factor_buy = 0
            std_deviations_buy = 0
            sma_period_buy = 0
            rsi_period_buy = 0
            rsi_low_buy = 0
            rsi_high_buy = 0
            max_hold_buy = 0
            min_operations_buy = 0
            slope_final_sell = 0
            intercept_final_sell = 0
            best_score_sell = 0
            expected_move_sell = 0
            loss_factor_sell = 0
            std_deviations_sell = 0
            sma_period_sell = 0
            rsi_period_sell = 0
            rsi_low_sell = 0
            rsi_high_sell = 0
            max_hold_sell = 0
            min_operations_sell = 0
            # get ticker history
            tick = yf.Ticker(ticker)
            ticker_data = tick.history(start=start_date, end=end_date, interval="1d")
            # count number of iterations
            noi = 0
            # COMBINE!
            for EXPECTED_MOVE in np.linspace(EXPECTED_MOVE_MIN, EXPECTED_MOVE_MAX, EXPECTED_MOVE_ITERATIONS):
                for LOSS_FACTOR in np.linspace(LOSS_FACTOR_MIN, LOSS_FACTOR_MAX, LOSS_FACTOR_ITERATIONS):
                    for STD_DEVIATIONS in np.linspace(STD_DEVIATIONS_MIN, STD_DEVIATIONS_MAX, STD_DEVIATIONS_ITERATIONS):
                        for SMA_PERIOD in range(SMA_PERIOD_MIN, SMA_PERIOD_MAX + SMA_PERIOD_STEP, SMA_PERIOD_STEP):
                            for RSI_PERIOD in range(RSI_PERIOD_MIN, RSI_PERIOD_MAX + RSI_PERIOD_STEP, RSI_PERIOD_STEP):
                                for RSI_LOW in range(RSI_LOW_MIN, RSI_LOW_MAX + RSI_LOW_STEP, RSI_LOW_STEP):
                                    for RSI_HIGH in range(RSI_HIGH_MIN, RSI_HIGH_MAX + RSI_HIGH_STEP, RSI_HIGH_STEP):
                                        for MAX_HOLD in range(MAX_HOLD_MIN, MAX_HOLD_MAX + MAX_HOLD_STEP, MAX_HOLD_STEP):
                                            for MIN_OPERATIONS_BUY in range(MIN_OPERATIONS_MIN_BUY, MIN_OPERATIONS_MAX_BUY + MIN_OPERATIONS_STEP, MIN_OPERATIONS_STEP):
                                                for MIN_OPERATIONS_SELL in range(MIN_OPERATIONS_MIN_SELL, MIN_OPERATIONS_MAX_SELL + MIN_OPERATIONS_STEP, MIN_OPERATIONS_STEP):
                                                    noi = noi + 1
                                                    print(noi)
                                                    # ----
                                                    # get data
                                                    # ----
                                                    bb_lower, bb_upper, rsi, prices_mean, prices_low, prices_high = self.mirai_no_data(
                                                            ticker_data=ticker_data,
                                                            start_date=start_date,
                                                            end_date=end_date,
                                                            STD_DEVIATIONS=STD_DEVIATIONS,
                                                            SMA_PERIOD=SMA_PERIOD,
                                                            RSI_PERIOD=RSI_PERIOD,
                                                            RSI_LOW=RSI_LOW,
                                                            RSI_HIGH=RSI_HIGH,
                                                            MAX_HOLD=MAX_HOLD,
                                                    )
                                                    n = len(bb_lower)
                                                    # ----
                                                    # select days to train and write their respective data
                                                    # ----
                                                    x_buy=[]
                                                    y_buy=[]
                                                    a_buy=[]
                                                    x_sell=[]
                                                    y_sell=[]
                                                    a_sell=[]
                                                    for i in range(0, n):
                                                        # sell
                                                        if prices_mean[i] > bb_upper[i] and rsi[i] > RSI_HIGH:
                                                            # take decision
                                                            decision = -1
                                                            for j in range(i + 1, min(i + MAX_HOLD, n)):
                                                                if prices_high[j] > (1 + EXPECTED_MOVE * LOSS_FACTOR) * prices_mean[i]:
                                                                    decision = 0
                                                                    break
                                                                elif prices_low[j] <= (1 - EXPECTED_MOVE) * prices_mean[i]:
                                                                    decision = 1
                                                                    break
                                                            if decision != -1:
                                                                x_sell.append(prices_mean[i] / bb_upper[i] - 1.0)
                                                                y_sell.append(rsi[i] / 100.0 - RSI_HIGH / 100.0)
                                                                a_sell.append(decision)

                                                        # buy
                                                        if prices_mean[i] < bb_lower[i] and rsi[i] < RSI_LOW:
                                                            decision = -1
                                                            for j in range(i + 1, min(i + MAX_HOLD, n)):
                                                                if prices_low[j] < (1 - EXPECTED_MOVE * LOSS_FACTOR) * prices_mean[i]:
                                                                    decision = 0
                                                                    break
                                                                if prices_high[j] >= (1 + EXPECTED_MOVE) * prices_mean[i]:
                                                                    decision = 1
                                                                    break
                                                            if decision != -1:
                                                                x_buy.append(bb_lower[i] / prices_mean[i] - 1.0)
                                                                y_buy.append((1.0 - rsi[i] / 100.0) - (1.0 - RSI_LOW / 100.0))
                                                                a_buy.append(decision)
                                                    slope_buy, intercept_buy, score_buy = self.mirai_no_yaiba(x_buy, y_buy, a_buy, MIN_OPERATIONS=MIN_OPERATIONS_BUY)
                                                    slope_sell, intercept_sell, score_sell = self.mirai_no_yaiba(x_sell, y_sell, a_sell, MIN_OPERATIONS=MIN_OPERATIONS_SELL)
                                                    if score_buy > best_score_buy:
                                                        '''
                                                        print("BUY")
                                                        plt.scatter(x_buy, y_buy,c=a_buy)
                                                        x_line = np.linspace(0,0.005,100)
                                                        y_line = []
                                                        for k in range(0,100):
                                                            y_line.append(slope_buy * x_line[k] + intercept_buy)
                                                        plt.plot(x_line, y_line)
                                                        plt.show()
                                                        '''
                                                        slope_final_buy = slope_buy
                                                        intercept_final_buy = intercept_buy
                                                        best_score_buy = score_buy
                                                        expected_move_buy = EXPECTED_MOVE
                                                        loss_factor_buy = LOSS_FACTOR
                                                        std_deviations_buy = STD_DEVIATIONS
                                                        sma_period_buy = SMA_PERIOD
                                                        rsi_period_buy = RSI_PERIOD
                                                        rsi_low_buy = RSI_LOW
                                                        rsi_high_buy = RSI_HIGH
                                                        max_hold_buy = MAX_HOLD
                                                        min_operations_buy = MIN_OPERATIONS_BUY

                                                    if score_sell > best_score_sell:
                                                        '''
                                                        print("SELL")
                                                        plt.scatter(x_sell, y_sell,c=a_sell)
                                                        x_line = np.linspace(0,0.005,100)
                                                        y_line = []
                                                        for k in range(0,100):
                                                            y_line.append(slope_sell * x_line[k] + intercept_sell)
                                                        plt.plot(x_line, y_line)
                                                        plt.show()
                                                        '''
                                                        slope_final_sell = slope_sell
                                                        intercept_final_sell = intercept_sell
                                                        best_score_sell = score_sell
                                                        expected_move_sell = EXPECTED_MOVE
                                                        loss_factor_sell = LOSS_FACTOR
                                                        std_deviations_sell = STD_DEVIATIONS
                                                        sma_period_sell = SMA_PERIOD
                                                        rsi_period_sell = RSI_PERIOD
                                                        rsi_low_sell = RSI_LOW
                                                        rsi_high_sell = RSI_HIGH
                                                        max_hold_sell = MAX_HOLD
                                                        min_operations_sell = MIN_OPERATIONS_SELL

            # write buy decision
            f = open("./data/" + ticker + "/sentaku_buy.txt", "w")
            if best_score_buy > 0:
                f.write(str(best_score_buy) + '\n')
                f.write(str(slope_final_buy) + '\n')
                f.write(str(intercept_final_buy) + '\n')
                f.write(str(expected_move_buy) + '\n')
                f.write(str(loss_factor_buy) + '\n')
                f.write(str(std_deviations_buy) + '\n')
                f.write(str(sma_period_buy) + '\n')
                f.write(str(rsi_period_buy) + '\n')
                f.write(str(rsi_low_buy) + '\n')
                f.write(str(rsi_high_buy) + '\n')
                f.write(str(max_hold_buy) + '\n')
                f.write(str(min_operations_buy) + '\n')
            else:
                f.write("n")
            f.close()
            # write sell decision
            f = open("./data/" + ticker + "/sentaku_sell.txt", "w")
            if best_score_sell > 0:
                f.write(str(best_score_sell) + '\n')
                f.write(str(slope_final_sell) + '\n')
                f.write(str(intercept_final_sell) + '\n')
                f.write(str(expected_move_sell) + '\n')
                f.write(str(loss_factor_sell) + '\n')
                f.write(str(std_deviations_sell) + '\n')
                f.write(str(sma_period_sell) + '\n')
                f.write(str(rsi_period_sell) + '\n')
                f.write(str(rsi_low_sell) + '\n')
                f.write(str(rsi_high_sell) + '\n')
                f.write(str(max_hold_sell) + '\n')
                f.write(str(min_operations_sell) + '\n')
            else:
                f.write("n")
            f.close()
            print(ticker + " DONE")

e = mirai_edge()
e.mirai_no_sentaku(date(2004,4,26), date(2022,4,1))
