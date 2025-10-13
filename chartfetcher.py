import os
import MetaTrader5 as mt5
import pandas as pd
import mplfinance as mpf
from datetime import datetime
import pytz
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Brokers configuration
brokersdictionary = {
    "deriv": {
        "TERMINAL_PATH": r"c:\xampp\htdocs\CIPHER\metaTrader5\cipher i\MetaTrader 5 deriv\terminal64.exe",
        "LOGIN_ID": "101347351",
        "PASSWORD": "@Techknowdge12#",
        "SERVER": "DerivSVG-Server-02",
        "ACCOUNT": "demo",
        "STRATEGY": "allorder",
        "BASE_FOLDER": r"C:\xampp\htdocs\CIPHER\cipher i\chart\cipher i\deriv\derivsymbols"
    },
    "deriv1": {
        "TERMINAL_PATH": r"c:\xampp\htdocs\CIPHER\metaTrader5\cipher i\MetaTrader 5 deriv 1\terminal64.exe",
        "LOGIN_ID": "140357859",
        "PASSWORD": "@Ayomide12#",
        "SERVER": "DerivSVG-Server-03",
        "ACCOUNT": "real",
        "STRATEGY": "lowtohigh",
        "BASE_FOLDER": r"C:\xampp\htdocs\CIPHER\cipher i\chart\cipher i\deriv 1\derivsymbols"
    },
    "deriv2": {
        "TERMINAL_PATH": r"c:\xampp\htdocs\CIPHER\metaTrader5\cipher i\MetaTrader 5 deriv 1\terminal64.exe",
        "LOGIN_ID": "140357853",
        "PASSWORD": "@Ayomide12#",
        "SERVER": "DerivSVG-Server-03",
        "ACCOUNT": "real",
        "STRATEGY": "hightolow",
        "BASE_FOLDER": r"C:\xampp\htdocs\CIPHER\cipher i\chart\cipher i\deriv 2\derivsymbols"
    },
    "bybit1": {
        "TERMINAL_PATH": r"c:\xampp\htdocs\CIPHER\metaTrader5\cipher i\MetaTrader 5 bybit 1\terminal64.exe",
        "LOGIN_ID": "4836528",
        "PASSWORD": "@Techknowdge12#",
        "SERVER": "Bybit-Live",
        "ACCOUNT": "real",
        "STRATEGY": "hightolow",
        "BASE_FOLDER": r"C:\xampp\htdocs\CIPHER\cipher i\chart\cipher i\bybit 1\bybitsymbols"
    },
    "exness1": {
        "TERMINAL_PATH": r"c:\xampp\htdocs\CIPHER\metaTrader5\cipher i\MetaTrader 5 exness 1\terminal64.exe",
        "LOGIN_ID": "63937917",
        "PASSWORD": "T1aiwoThrone2",
        "SERVER": "Exness-MT5Real9",
        "ACCOUNT": "real",
        "STRATEGY": "hightolow",
        "BASE_FOLDER": r"C:\xampp\htdocs\CIPHER\cipher i\chart\cipher i\exness 1\exness_symbols"
    }
}

BASE_ERROR_FOLDER = r"C:\xampp\htdocs\CIPHER\cipher i\chart\cipher i\debugs"
TIMEFRAME_MAP = {
    "5m": mt5.TIMEFRAME_M5,
    "15m": mt5.TIMEFRAME_M15,
    "30m": mt5.TIMEFRAME_M30,
    "1h": mt5.TIMEFRAME_H1,
    "4h": mt5.TIMEFRAME_H4
}
ERROR_JSON_PATH = os.path.join(BASE_ERROR_FOLDER, "chart_errors.json")
           
def clear_chart_folder(base_folder):
    """Clear all contents of the chart folder to ensure fresh data is saved."""
    error_log = []
    try:
        if not os.path.exists(base_folder):
            log_and_print(f"Chart folder {base_folder} does not exist, no need to clear.", "INFO")
            return True, error_log

        for item in os.listdir(base_folder):
            item_path = os.path.join(base_folder, item)
            try:
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    import shutil
                    shutil.rmtree(item_path)
                log_and_print(f"Deleted {item_path}", "INFO")
            except Exception as e:
                error_log.append({
                    "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                    "error": f"Failed to delete {item_path}: {str(e)}",
                    "broker": base_folder
                })
                log_and_print(f"Failed to delete {item_path}: {str(e)}", "ERROR")

        log_and_print(f"Chart folder {base_folder} cleared successfully", "SUCCESS")
        return True, error_log
    except Exception as e:
        error_log.append({
            "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
            "error": f"Failed to clear chart folder {base_folder}: {str(e)}",
            "broker": base_folder
        })
        save_errors(error_log)
        log_and_print(f"Failed to clear chart folder {base_folder}: {str(e)}", "ERROR")
        return False, error_log

def log_and_print(message, level="INFO"):
    """Log and print messages in a structured format."""
    timestamp = datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {level:8} | {message}")

def save_errors(error_log):
    """Save error log to JSON file."""
    try:
        os.makedirs(BASE_ERROR_FOLDER, exist_ok=True)
        with open(ERROR_JSON_PATH, 'w') as f:
            json.dump(error_log, f, indent=4)
        log_and_print("Error log saved", "ERROR")
    except Exception as e:
        log_and_print(f"Failed to save error log: {str(e)}", "ERROR")

def initialize_mt5(terminal_path, login_id, password, server):
    """Initialize MetaTrader 5 terminal for a specific broker."""
    error_log = []
    if not os.path.exists(terminal_path):
        error_log.append({
            "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
            "error": f"MT5 terminal executable not found: {terminal_path}",
            "broker": server
        })
        save_errors(error_log)
        log_and_print(f"MT5 terminal executable not found: {terminal_path}", "ERROR")
        return False, error_log

    try:
        if not mt5.initialize(
            path=terminal_path,
            login=int(login_id),
            server=server,
            password=password,
            timeout=30000
        ):
            error_log.append({
                "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                "error": f"Failed to initialize MT5: {mt5.last_error()}",
                "broker": server
            })
            save_errors(error_log)
            log_and_print(f"Failed to initialize MT5: {mt5.last_error()}", "ERROR")
            return False, error_log

        if not mt5.login(login=int(login_id), server=server, password=password):
            error_log.append({
                "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                "error": f"Failed to login to MT5: {mt5.last_error()}",
                "broker": server
            })
            save_errors(error_log)
            log_and_print(f"Failed to login to MT5: {mt5.last_error()}", "ERROR")
            mt5.shutdown()
            return False, error_log

        log_and_print(f"MT5 initialized and logged in successfully (loginid={login_id}, server={server})", "SUCCESS")
        return True, error_log
    except Exception as e:
        error_log.append({
            "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
            "error": f"Unexpected error in initialize_mt5: {str(e)}",
            "broker": server
        })
        save_errors(error_log)
        log_and_print(f"Unexpected error in initialize_mt5: {str(e)}", "ERROR")
        return False, error_log

def get_symbols():
    """Retrieve all available symbols from MT5."""
    error_log = []
    symbols = mt5.symbols_get()
    if not symbols:
        error_log.append({
            "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
            "error": f"Failed to retrieve symbols: {mt5.last_error()}",
            "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
        })
        save_errors(error_log)
        log_and_print(f"Failed to retrieve symbols: {mt5.last_error()}", "ERROR")
        return [], error_log

    available_symbols = [s.name for s in symbols]
    log_and_print(f"Retrieved {len(available_symbols)} symbols", "INFO")
    return available_symbols, error_log

def fetch_ohlcv_data(symbol, mt5_timeframe, bars):
    """Fetch OHLCV data for a given symbol and timeframe."""
    error_log = []
    if not mt5.symbol_select(symbol, True):
        error_log.append({
            "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
            "error": f"Failed to select symbol {symbol}: {mt5.last_error()}",
            "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
        })
        save_errors(error_log)
        log_and_print(f"Failed to select symbol {symbol}: {mt5.last_error()}", "ERROR")
        return None, error_log

    rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, bars)
    if rates is None or len(rates) == 0:
        error_log.append({
            "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
            "error": f"Failed to retrieve rates for {symbol}: {mt5.last_error()}",
            "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
        })
        save_errors(error_log)
        log_and_print(f"Failed to retrieve rates for {symbol}: {mt5.last_error()}", "ERROR")
        return None, error_log

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time")
    df = df.astype({
        "open": float, "high": float, "low": float, "close": float,
        "tick_volume": float, "spread": int, "real_volume": float
    })
    df.rename(columns={"tick_volume": "volume"}, inplace=True)
    log_and_print(f"OHLCV data fetched for {symbol}", "INFO")
    return df, error_log

def identifyparenthighsandlows(df, neighborcandles_left, neighborcandles_right):
    """Identify Parent Highs (PH) and Parent Lows (PL) based on neighbor candles."""
    error_log = []
    ph_indices = []
    pl_indices = []
    ph_labels = []
    pl_labels = []

    try:
        for i in range(len(df)):
            if i >= len(df) - neighborcandles_right:
                continue

            current_high = df.iloc[i]['high']
            current_low = df.iloc[i]['low']
            right_highs = df.iloc[i + 1:i + neighborcandles_right + 1]['high']
            right_lows = df.iloc[i + 1:i + neighborcandles_right + 1]['low']
            left_highs = df.iloc[max(0, i - neighborcandles_left):i]['high']
            left_lows = df.iloc[max(0, i - neighborcandles_left):i]['low']

            if len(right_highs) == neighborcandles_right:
                is_ph = True
                if len(left_highs) > 0:
                    is_ph = current_high > left_highs.max()
                is_ph = is_ph and current_high > right_highs.max()
                if is_ph:
                    ph_indices.append(df.index[i])
                    ph_labels.append(('PH', current_high, df.index[i]))

            if len(right_lows) == neighborcandles_right:
                is_pl = True
                if len(left_lows) > 0:
                    is_pl = current_low < left_lows.min()
                is_pl = is_pl and current_low < right_lows.min()
                if is_pl:
                    pl_indices.append(df.index[i])
                    pl_labels.append(('PL', current_low, df.index[i]))

        log_and_print(f"Identified {len(ph_indices)} PH and {len(pl_indices)} PL for {df['symbol'].iloc[0]}", "INFO")
        return ph_labels, pl_labels, error_log
    except Exception as e:
        error_log.append({
            "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
            "error": f"Failed to identify PH/PL: {str(e)}",
            "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
        })
        save_errors(error_log)
        log_and_print(f"Failed to identify PH/PL: {str(e)}", "ERROR")
        return [], [], error_log

def save_candle_data(df, symbol, timeframe_str, timeframe_folder, ph_labels, pl_labels):
    """Save all candle data with numbering and PH/PL labels."""
    error_log = []
    candle_json_path = os.path.join(timeframe_folder, "all_candles.json")
    try:
        if len(df) >= 2:
            candles = []
            ph_dict = {t: label for label, _, t in ph_labels}
            pl_dict = {t: label for label, _, t in pl_labels}

            for i, (index, row) in enumerate(df[::-1].iterrows()):
                candle = row.to_dict()
                candle["time"] = index.strftime('%Y-%m-%d %H:%M:%S')
                candle["candle_number"] = i
                candle["symbol"] = symbol
                candle["timeframe"] = timeframe_str
                candle["is_ph"] = ph_dict.get(index, None) == 'PH'
                candle["is_pl"] = pl_dict.get(index, None) == 'PL'
                candles.append(candle)
            with open(candle_json_path, 'w') as f:
                json.dump(candles, f, indent=4)
            log_and_print(f"Candle data saved for {symbol} ({timeframe_str})", "SUCCESS")
        else:
            error_log.append({
                "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                "error": f"Not enough data to save candles for {symbol} ({timeframe_str})",
                "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
            })
            save_errors(error_log)
            log_and_print(f"Not enough data to save candles for {symbol} ({timeframe_str})", "ERROR")
    except Exception as e:
        error_log.append({
            "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
            "error": f"Failed to save candles for {symbol} ({timeframe_str}): {str(e)}",
            "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
        })
        save_errors(error_log)
        log_and_print(f"Failed to save candles for {symbol} ({timeframe_str}): {str(e)}", "ERROR")
    return error_log

def generate_and_save_chart(df, symbol, timeframe_str, timeframe_folder, neighborcandles_left, neighborcandles_right):
    """Generate and save a basic candlestick chart as chart.png, then identify PH/PL and save as chartanalysed.png with markers."""
    error_log = []
    chart_path = os.path.join(timeframe_folder, "chart.png")
    chart_analysed_path = os.path.join(timeframe_folder, "chartanalysed.png")
    trendline_log_json_path = os.path.join(timeframe_folder, "trendline_log.json")
    trendline_log = []

    try:
        custom_style = mpf.make_mpf_style(
            base_mpl_style="default",
            marketcolors=mpf.make_marketcolors(
                up="green",
                down="red",
                edge="inherit",
                wick={"up": "green", "down": "red"},
                volume="gray"
            )
        )

        # Step 1: Save basic candlestick chart as chart.png
        fig, axlist = mpf.plot(
            df,
            type='candle',
            style=custom_style,
            volume=False,
            title=f"{symbol} ({timeframe_str})",
            returnfig=True
        )

        # Adjust wick thickness for basic chart
        for ax in axlist:
            for line in ax.get_lines():
                if line.get_label() == '':
                    line.set_linewidth(0.5)

        current_size = fig.get_size_inches()
        fig.set_size_inches(25, current_size[1])
        axlist[0].grid(False)
        fig.savefig(chart_path, bbox_inches="tight", dpi=100)
        plt.close(fig)
        log_and_print(f"Basic chart saved for {symbol} ({timeframe_str}) as {chart_path}", "SUCCESS")

        # Step 2: Identify PH/PL
        ph_labels, pl_labels, phpl_errors = identifyparenthighsandlows(df, neighborcandles_left, neighborcandles_right)
        error_log.extend(phpl_errors)

        # Step 3: Prepare annotations for analyzed chart with PH/PL markers
        apds = []
        if ph_labels:
            ph_series = pd.Series([np.nan] * len(df), index=df.index)
            for _, price, t in ph_labels:
                ph_series.loc[t] = price
            apds.append(mpf.make_addplot(
                ph_series,
                type='scatter',
                markersize=100,
                marker='^',
                color='blue'
            ))
        if pl_labels:
            pl_series = pd.Series([np.nan] * len(df), index=df.index)
            for _, price, t in pl_labels:
                pl_series.loc[t] = price
            apds.append(mpf.make_addplot(
                pl_series,
                type='scatter',
                markersize=100,
                marker='v',
                color='purple'
            ))

        trendline_log.append({
            "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
            "symbol": symbol,
            "timeframe": timeframe_str,
            "team_type": "initial",
            "status": "info",
            "reason": f"Found {len(ph_labels)} PH points and {len(pl_labels)} PL points",
            "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
        })

        # Save Trendline Log (only PH/PL info, no trendlines)
        try:
            with open(trendline_log_json_path, 'w') as f:
                json.dump(trendline_log, f, indent=4)
            log_and_print(f"Trendline log saved for {symbol} ({timeframe_str})", "SUCCESS")
        except Exception as e:
            error_log.append({
                "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                "error": f"Failed to save trendline log for {symbol} ({timeframe_str}): {str(e)}",
                "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
            })
            log_and_print(f"Failed to save trendline log for {symbol} ({timeframe_str}): {str(e)}", "ERROR")

        # Step 4: Save analyzed chart with PH/PL markers as chartanalysed.png
        fig, axlist = mpf.plot(
            df,
            type='candle',
            style=custom_style,
            volume=False,
            title=f"{symbol} ({timeframe_str}) - Analysed",
            addplot=apds if apds else None,
            returnfig=True
        )

        # Adjust wick thickness for analyzed chart
        for ax in axlist:
            for line in ax.get_lines():
                if line.get_label() == '':
                    line.set_linewidth(0.5)

        current_size = fig.get_size_inches()
        fig.set_size_inches(25, current_size[1])
        axlist[0].grid(True, linestyle='--')
        fig.savefig(chart_analysed_path, bbox_inches="tight", dpi=100)
        plt.close(fig)
        log_and_print(f"Analysed chart saved for {symbol} ({timeframe_str}) as {chart_analysed_path}", "SUCCESS")

        return chart_path, error_log, ph_labels, pl_labels
    except Exception as e:
        error_log.append({
            "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
            "error": f"Failed to save charts for {symbol} ({timeframe_str}): {str(e)}",
            "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
        })
        trendline_log.append({
            "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
            "symbol": symbol,
            "timeframe": timeframe_str,
            "status": "failed",
            "reason": f"Chart generation failed: {str(e)}",
            "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
        })
        with open(trendline_log_json_path, 'w') as f:
            json.dump(trendline_log, f, indent=4)
        save_errors(error_log)
        log_and_print(f"Failed to save charts for {symbol} ({timeframe_str}): {str(e)}", "ERROR")
        return chart_path if os.path.exists(chart_path) else None, error_log, [], []

def draw_dashed_line(img, start_point, end_point, color=(0, 0, 0), thickness=1, dash_length=10, gap_length=10):
    """Draw a dashed line on the image from start_point to end_point."""
    x1, y1 = start_point
    x2, y2 = end_point
    # Calculate the length of the line
    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    if length == 0:
        return
    # Calculate the unit vector
    dx = (x2 - x1) / length
    dy = (y2 - y1) / length
    # Draw dashes
    current_length = 0
    while current_length < length:
        start_x = int(x1 + dx * current_length)
        start_y = int(y1 + dy * current_length)
        end_x = int(x1 + dx * min(current_length + dash_length, length))
        end_y = int(y1 + dy * min(current_length + dash_length, length))
        cv2.line(img, (start_x, start_y), (end_x, end_y), color, thickness)
        current_length += dash_length + gap_length

def detect_candle_contours(chart_path, symbol, timeframe_str, timeframe_folder, candleafterintersector=2, minbreakoutcandleposition=5):
    error_log = []
    contour_json_path = os.path.join(timeframe_folder, "chart_contours.json")
    trendline_log_json_path = os.path.join(timeframe_folder, "trendline_log.json")
    output_image_path = os.path.join(timeframe_folder, "chart_with_contours.png")
    candle_json_path = os.path.join(timeframe_folder, "all_candles.json")
    trendline_log = []

    def draw_chevron_arrow(img, x, y, direction, color=(0, 0, 255), line_length=15, chevron_size=8):
        """
        Draw a chevron-style arrow on the image at position (x, y).
        direction: 'up' for upward chevron (base at y, chevron '^' at y - line_length),
                  'down' for downward chevron (base at y, chevron 'v' at y + line_length).
        color: BGR tuple, default red (0, 0, 255).
        line_length: Length of the vertical line in pixels.
        chevron_size: Width of the chevron head (distance from center to each wing tip) in pixels.
        """
        if direction == 'up':
            top_y = y - line_length
            cv2.line(img, (x, y), (x, top_y), color, thickness=1)
            cv2.line(img, (x - chevron_size // 2, top_y + chevron_size // 2), (x, top_y), color, thickness=1)
            cv2.line(img, (x + chevron_size // 2, top_y + chevron_size // 2), (x, top_y), color, thickness=1)
        else:  # direction == 'down'
            top_y = y + line_length
            cv2.line(img, (x, y), (x, top_y), color, thickness=1)
            cv2.line(img, (x - chevron_size // 2, top_y - chevron_size // 2), (x, top_y), color, thickness=1)
            cv2.line(img, (x + chevron_size // 2, top_y - chevron_size // 2), (x, top_y), color, thickness=1)

    try:
        img = cv2.imread(chart_path)
        if img is None:
            error_log.append({
                "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                "error": f"Failed to load chart image {chart_path} for contour detection",
                "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
            })
            save_errors(error_log)
            log_and_print(f"Failed to load chart image {chart_path} for contour detection", "ERROR")
            return error_log

        img_height, img_width = img.shape[:2]

        try:
            with open(candle_json_path, 'r') as f:
                candle_data = json.load(f)
        except Exception as e:
            error_log.append({
                "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                "error": f"Failed to load {candle_json_path} for PH/PL data: {str(e)}",
                "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
            })
            save_errors(error_log)
            log_and_print(f"Failed to load {candle_json_path} for PH/PL data: {str(e)}", "ERROR")
            return error_log

        ph_candles = [c for c in candle_data if c.get("is_ph", False)]
        pl_candles = [c for c in candle_data if c.get("is_pl", False)]
        ph_indices = {int(c["candle_number"]): c for c in ph_candles}
        pl_indices = {int(c["candle_number"]): c for c in pl_candles}

        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        green_lower = np.array([35, 50, 50])
        green_upper = np.array([85, 255, 255])
        green_mask = cv2.inRange(img_hsv, green_lower, green_upper)
        red_lower1 = np.array([0, 50, 50])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 50, 50])
        red_upper2 = np.array([180, 255, 255])
        red_mask = cv2.inRange(img_hsv, red_lower1, red_upper1) | cv2.inRange(img_hsv, red_lower2, red_upper2)
        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        green_contours = sorted(green_contours, key=lambda c: cv2.boundingRect(c)[0], reverse=True)
        red_contours = sorted(red_contours, key=lambda c: cv2.boundingRect(c)[0], reverse=True)
        green_count = len(green_contours)
        red_count = len(red_contours)
        total_count = green_count + red_count
        all_contours = green_contours + red_contours
        all_contours = sorted(all_contours, key=lambda c: cv2.boundingRect(c)[0], reverse=True)

        contour_positions = {}
        candle_bounds = {}
        for i, contour in enumerate(all_contours):
            x, y, w, h = cv2.boundingRect(contour)
            contour_positions[i] = {"x": x + w // 2, "y": y, "width": w, "height": h}
            candle = candle_data[i]
            candle_bounds[i] = {
                "high_y": y,
                "low_y": y + h,
                "body_top_y": y + min(h // 4, 10),
                "body_bottom_y": y + h - min(h // 4, 10),
                "x_left": x,
                "x_right": x + w,
                "high": float(candle["high"]),
                "low": float(candle["low"])
            }

        for i, contour in enumerate(all_contours):
            x, y, w, h = cv2.boundingRect(contour)
            if cv2.pointPolygonTest(contour, (x + w // 2, y + h // 2), False) >= 0:
                if green_mask[y + h // 2, x + w // 2] > 0:
                    cv2.drawContours(img, [contour], -1, (0, 128, 0), 1)
                elif red_mask[y + h // 2, x + w // 2] > 0:
                    cv2.drawContours(img, [contour], -1, (0, 0, 255), 1)
            if i in ph_indices:
                points = np.array([
                    [x + w // 2, y - 10],
                    [x + w // 2 - 10, y + 5],
                    [x + w // 2 + 10, y + 5]
                ])
                cv2.fillPoly(img, [points], color=(255, 0, 0))
            if i in pl_indices:
                points = np.array([
                    [x + w // 2, y + h + 10],
                    [x + w // 2 - 10, y + h - 5],
                    [x + w // 2 + 10, y + h - 5]
                ])
                cv2.fillPoly(img, [points], color=(128, 0, 128))

        def find_intersectors(sender_idx, receiver_idx, sender_x, sender_y, is_ph, receiver_price):
            intersectors = []
            receiver_pos = contour_positions.get(receiver_idx)
            receiver_y = receiver_pos["y"] if is_ph else receiver_pos["y"] + receiver_pos["height"]
            dx = receiver_pos["x"] - sender_x
            dy = receiver_y - sender_y
            if dx == 0:
                slope = float('inf')
            else:
                slope = dy / dx
            i = receiver_idx - 1
            found_first = False
            first_intersector_price = None
            previous_intersector_price = None
            while i >= 0:
                if i not in candle_bounds or i == sender_idx or i == receiver_idx:
                    i -= 1
                    continue
                bounds = candle_bounds[i]
                x = bounds["x_left"]
                if slope == float('inf'):
                    y = sender_y
                else:
                    y = sender_y + slope * (x - sender_x)
                if not found_first:
                    price = bounds["high"] if is_ph else bounds["low"]
                    price_range = max(c["high"] for c in candle_data) - min(c["low"] for c in candle_data)
                    if price_range == 0:
                        y_price = y
                    else:
                        min_y = min(b["high_y"] for b in candle_bounds.values())
                        max_y = max(b["low_y"] for b in candle_bounds.values())
                        price_min = min(c["low"] for c in candle_data)
                        y_price = max_y - ((price - price_min) / price_range) * (max_y - min_y)
                        y_price = int(y_price)
                    if abs(y - y_price) <= 10:
                        check_idx = i - candleafterintersector
                        is_trendbreaker = False
                        if check_idx >= 0 and check_idx in candle_bounds:
                            check_candle = candle_bounds[check_idx]
                            check_price = check_candle["high"] if is_ph else check_candle["low"]
                            if (is_ph and check_price > receiver_price) or (not is_ph and check_price < receiver_price):
                                is_trendbreaker = True
                        intersectors.append((i, x, y_price, price, True, bounds["high_y"], bounds["low_y"], is_trendbreaker))
                        found_first = True
                        first_intersector_price = price
                        previous_intersector_price = price
                        if is_trendbreaker:
                            break
                        i -= 1
                        continue
                if found_first and bounds["body_top_y"] <= y <= bounds["body_bottom_y"]:
                    current_price = bounds["high"] if is_ph else bounds["low"]
                    is_trendbreaker = False
                    check_idx = i - candleafterintersector
                    if check_idx >= 0 and check_idx in candle_bounds:
                        check_candle = candle_bounds[check_idx]
                        check_price = check_candle["high"] if is_ph else check_candle["low"]
                        if (is_ph and check_price > previous_intersector_price) or (not is_ph and check_price < previous_intersector_price):
                            is_trendbreaker = True
                        elif is_ph and current_price > first_intersector_price:
                            is_trendbreaker = True
                        elif not is_ph and current_price < first_intersector_price:
                            is_trendbreaker = True
                    intersectors.append((i, x, int(y), None, False, bounds["high_y"], bounds["low_y"], is_trendbreaker))
                    previous_intersector_price = current_price
                    if is_trendbreaker:
                        break
                i -= 1
            return intersectors, slope

        def find_intruder(sender_idx, receiver_idx, sender_price, is_ph):
            start_idx = min(sender_idx, receiver_idx) + 1
            end_idx = max(sender_idx, receiver_idx) - 1
            for i in range(start_idx, end_idx + 1):
                if i in candle_bounds:
                    candle = candle_bounds[i]
                    price = candle["high"] if is_ph else candle["low"]
                    if (is_ph and price > sender_price) or (not is_ph and price < sender_price):
                        return i
            return None

        def find_first_body_intersection(start_idx, start_y, price, is_ph):
            for idx in range(start_idx - 1, -1, -1):
                if idx not in candle_bounds:
                    continue
                bounds = candle_bounds[idx]
                if is_ph:
                    if bounds["body_top_y"] <= start_y <= bounds["body_bottom_y"]:
                        return idx, bounds["x_left"], start_y
                else:
                    if bounds["body_top_y"] <= start_y <= bounds["body_bottom_y"]:
                        return idx, bounds["x_left"], start_y
            return None

        # Process PH-to-PH trendlines
        ph_teams = []
        pl_teams = []
        ph_additional_trendlines = []
        pl_additional_trendlines = []
        sorted_ph = sorted(ph_indices.items(), key=lambda x: x[0], reverse=True)
        sorted_pl = sorted(pl_indices.items(), key=lambda x: x[0], reverse=True)

        i = 0
        while i < len(sorted_ph) - 1:
            sender_idx, sender_data = sorted_ph[i]
            sender_high = float(sender_data["high"])
            if i + 1 < len(sorted_ph):
                next_idx, next_data = sorted_ph[i + 1]
                next_high = float(next_data["high"])
                if next_high > sender_high:
                    trendline_log.append({
                        "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                        "symbol": symbol,
                        "timeframe": timeframe_str,
                        "team_type": "PH-to-PH",
                        "status": "skipped",
                        "reason": f"Immediate intruder PH found at candle {next_idx} (high {next_high}) higher than sender high {sender_high} (candle {sender_idx}), setting intruder as new sender",
                        "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
                    })
                    i += 1
                    continue
            best_receiver_idx = None
            best_receiver_high = float('-inf')
            j = i + 1
            while j < len(sorted_ph):
                candidate_idx, candidate_data = sorted_ph[j]
                candidate_high = float(candidate_data["high"])
                if sender_high > candidate_high > best_receiver_high:
                    best_receiver_idx = candidate_idx
                    best_receiver_high = candidate_high
                j += 1
            if best_receiver_idx is not None:
                intruder_idx = find_intruder(sender_idx, best_receiver_idx, sender_high, is_ph=True)
                if intruder_idx is not None:
                    intruder_high = candle_bounds[intruder_idx]["high"]
                    trendline_log.append({
                        "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                        "symbol": symbol,
                        "timeframe": timeframe_str,
                        "team_type": "PH-to-PH",
                        "status": "skipped",
                        "reason": f"Intruder candle found at candle {intruder_idx} (high {intruder_high}) higher than sender high {sender_high} (candle {sender_idx}) between sender and receiver (candle {best_receiver_idx}), setting receiver as new sender",
                        "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
                    })
                    i = next((k for k, (idx, _) in enumerate(sorted_ph) if idx == best_receiver_idx), i + 1)
                    continue
                sender_pos = contour_positions.get(sender_idx)
                receiver_pos = contour_positions.get(best_receiver_idx)
                if sender_pos and receiver_pos:
                    sender_x = sender_pos["x"]
                    sender_y = sender_pos["y"]
                    receiver_x = receiver_pos["x"]
                    receiver_y = receiver_pos["y"]
                    intersectors, slope = find_intersectors(sender_idx, best_receiver_idx, sender_x, sender_y, is_ph=True, receiver_price=best_receiver_high)
                    team_data = {
                        "sender": {"candle_number": sender_idx, "high": sender_high, "x": sender_x, "y": sender_y},
                        "receiver": {"candle_number": best_receiver_idx, "high": best_receiver_high, "x": receiver_x, "y": receiver_y},
                        "intersectors": [],
                        "trendlines": []
                    }
                    reason = ""
                    selected_idx = best_receiver_idx
                    selected_x = receiver_x
                    selected_y = receiver_y
                    selected_high = best_receiver_high
                    selected_type = "receiver"
                    selected_is_first = False
                    selected_marker = None
                    min_high = best_receiver_high
                    for idx, x, y, price, is_first, high_y, low_y, is_trendbreaker in intersectors:
                        marker = "star" if is_first else "circle"
                        team_data["intersectors"].append({
                            "candle_number": idx,
                            "high": price if is_first else None,
                            "x": x,
                            "y": high_y,
                            "is_first": is_first,
                            "is_trendbreaker": is_trendbreaker,
                            "marker": marker
                        })
                        if is_trendbreaker:
                            reason += f"; detected {'first' if is_first else 'subsequent'} intersector (candle {idx}, high {price if is_first else candle_bounds[idx]['high']}, x={x}, y={high_y}, marker={marker}, trendbreaker=True), skipped trendline generation"
                        else:
                            current_high = price if is_first else candle_bounds[idx]["high"]
                            if current_high < min_high:
                                min_high = current_high
                                selected_idx = idx
                                selected_x = x
                                selected_y = high_y
                                selected_high = current_high
                                selected_type = "intersector"
                                selected_is_first = is_first
                                selected_marker = "star" if is_first else "circle"
                            reason += f"; detected {'first' if is_first else 'subsequent'} intersector (candle {idx}, high {price if is_first else candle_bounds[idx]['high']}, x={x}, y={high_y}, marker={marker}, trendbreaker=False)"
                    # Draw trendline and check breakout only for the selected trendline
                    if selected_type == "receiver" and not intersectors:
                        # Find first breakout candle
                        first_breakout_idx = None
                        for check_idx in range(selected_idx - 1, -1, -1):
                            if check_idx in candle_bounds:
                                next_candle = candle_bounds[check_idx]
                                if (next_candle["low"] > candle_bounds[selected_idx]["low"] and
                                    next_candle["high"] > candle_bounds[selected_idx]["high"]):
                                    first_breakout_idx = check_idx
                                    break
                        # Extend trendline to first breakout candle if found
                        end_x = selected_x
                        end_y = selected_y
                        if first_breakout_idx is not None and first_breakout_idx in candle_bounds:
                            end_x = candle_bounds[first_breakout_idx]["x_left"]
                            if slope != float('inf'):
                                end_y = int(sender_y + slope * (end_x - sender_x))
                            else:
                                end_y = sender_y
                            reason += f"; extended trendline to x-axis of first breakout candle {first_breakout_idx} (x={end_x}, y={end_y})"
                        cv2.line(img, (sender_x, sender_y), (end_x, end_y), color=(255, 0, 0), thickness=1)
                        team_data["trendlines"].append({
                            "type": "receiver",
                            "candle_number": selected_idx,
                            "high": selected_high,
                            "x": end_x,
                            "y": end_y
                        })
                        ph_additional_trendlines.append({
                            "type": "receiver",
                            "candle_number": selected_idx,
                            "high": selected_high,
                            "x": end_x,
                            "y": end_y
                        })
                        reason = f"Drew PH-to-PH trendline from sender (candle {sender_idx}, high {sender_high}, x={sender_x}, y={sender_y}) to receiver (candle {selected_idx}, high {selected_high}, x={end_x}, y={end_y}) as no intersectors found" + reason
                        # Check for breakout candle for chevron arrow
                        if first_breakout_idx is not None:
                            start_idx = max(0, first_breakout_idx - minbreakoutcandleposition)
                            for check_idx in range(start_idx, -1, -1):
                                if check_idx in candle_bounds:
                                    next_candle = candle_bounds[check_idx]
                                    if (next_candle["low"] > candle_bounds[selected_idx]["low"] and
                                        next_candle["high"] > candle_bounds[selected_idx]["high"]):
                                        target_idx = check_idx
                                        target_x = candle_bounds[target_idx]["x_left"]
                                        target_y = candle_bounds[target_idx]["high_y"] + 10  # Add offset for PH
                                        draw_chevron_arrow(img, target_x + (candle_bounds[target_idx]["x_right"] - candle_bounds[target_idx]["x_left"]) // 2, target_y, 'up', color=(255, 0, 0))
                                        reason += f"; for selected trendline to candle {selected_idx}, found first breakout candle {first_breakout_idx} (high={candle_bounds[first_breakout_idx]['high']}, low={candle_bounds[first_breakout_idx]['low']}), skipped {minbreakoutcandleposition} candles, selected target candle {target_idx} (high_y={target_y}, high={candle_bounds[target_idx]['high']}, low={candle_bounds[target_idx]['low']}) for blue chevron arrow at center x-axis with offset from high"
                                        break
                    elif selected_type == "intersector":
                        # Find first breakout candle
                        first_breakout_idx = None
                        for check_idx in range(selected_idx - 1, -1, -1):
                            if check_idx in candle_bounds:
                                next_candle = candle_bounds[check_idx]
                                if (next_candle["low"] > candle_bounds[selected_idx]["low"] and
                                    next_candle["high"] > candle_bounds[selected_idx]["high"]):
                                    first_breakout_idx = check_idx
                                    break
                        # Extend trendline to first breakout candle if found
                        end_x = selected_x
                        end_y = selected_y
                        if first_breakout_idx is not None and first_breakout_idx in candle_bounds:
                            end_x = candle_bounds[first_breakout_idx]["x_left"]
                            if slope != float('inf'):
                                end_y = int(sender_y + slope * (end_x - sender_x))
                            else:
                                end_y = sender_y
                            reason += f"; extended trendline to x-axis of first breakout candle {first_breakout_idx} (x={end_x}, y={end_y})"
                        cv2.line(img, (sender_x, sender_y), (end_x, end_y), color=(255, 0, 0), thickness=1)
                        if selected_is_first:
                            star_points = [
                                [selected_x, selected_y - 15], [selected_x + 4, selected_y - 5], [selected_x + 14, selected_y - 5],
                                [selected_x + 5, selected_y + 2], [selected_x + 10, selected_y + 12], [selected_x, selected_y + 7],
                                [selected_x - 10, selected_y + 12], [selected_x - 5, selected_y + 2], [selected_x - 14, selected_y - 5],
                                [selected_x - 4, selected_y - 5]
                            ]
                            cv2.fillPoly(img, [np.array(star_points)], color=(255, 0, 0))
                        else:
                            cv2.circle(img, (selected_x, selected_y), 5, color=(255, 0, 0), thickness=-1)
                        team_data["trendlines"].append({
                            "type": "intersector",
                            "candle_number": selected_idx,
                            "high": selected_high,
                            "x": end_x,
                            "y": end_y,
                            "is_first": selected_is_first,
                            "is_trendbreaker": False,
                            "marker": selected_marker
                        })
                        ph_additional_trendlines.append({
                            "type": "intersector",
                            "candle_number": selected_idx,
                            "high": selected_high,
                            "x": end_x,
                            "y": end_y,
                            "is_first": selected_is_first,
                            "is_trendbreaker": False,
                            "marker": selected_marker
                        })
                        reason = f"Drew PH-to-PH trendline from sender (candle {sender_idx}, high {sender_high}, x={sender_x}, y={sender_y}) to intersector (candle {selected_idx}, high {selected_high}, x={end_x}, y={end_y}, marker={selected_marker}, trendbreaker=False) with lowest high" + reason
                        # Check for breakout candle for chevron arrow
                        if first_breakout_idx is not None:
                            start_idx = max(0, first_breakout_idx - minbreakoutcandleposition)
                            for check_idx in range(start_idx, -1, -1):
                                if check_idx in candle_bounds:
                                    next_candle = candle_bounds[check_idx]
                                    if (next_candle["low"] > candle_bounds[selected_idx]["low"] and
                                        next_candle["high"] > candle_bounds[selected_idx]["high"]):
                                        target_idx = check_idx
                                        target_x = candle_bounds[target_idx]["x_left"]
                                        target_y = candle_bounds[target_idx]["high_y"] + 10  # Add offset for PH
                                        draw_chevron_arrow(img, target_x + (candle_bounds[target_idx]["x_right"] - target_x) // 2, target_y, 'up', color=(255, 0, 0))
                                        reason += f"; for selected trendline to candle {selected_idx}, found first breakout candle {first_breakout_idx} (high={candle_bounds[first_breakout_idx]['high']}, low={candle_bounds[first_breakout_idx]['low']}), skipped {minbreakoutcandleposition} candles, selected target candle {target_idx} (high_y={target_y}, high={candle_bounds[target_idx]['high']}, low={candle_bounds[target_idx]['low']}) for blue chevron arrow at center x-axis with offset from high"
                                        break
                    else:
                        reason = f"No PH-to-PH trendline drawn from sender (candle {sender_idx}, high {sender_high}, x={sender_x}, y={sender_y}) as first intersector is trendbreaker" + reason
                    trendline_log.append({
                        "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                        "symbol": symbol,
                        "timeframe": timeframe_str,
                        "team_type": "PH-to-PH",
                        "status": "success" if team_data["trendlines"] else "skipped",
                        "reason": reason,
                        "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
                    })
                    ph_teams.append(team_data)
                    i = next((k for k, (idx, _) in enumerate(sorted_ph) if idx == best_receiver_idx), i + 1) + 1
                else:
                    error_log.append({
                        "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                        "error": f"Missing contour positions for PH sender (candle {sender_idx}) or receiver (candle {best_receiver_idx})",
                        "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
                    })
                    i += 1
            else:
                trendline_log.append({
                    "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                    "symbol": symbol,
                    "timeframe": timeframe_str,
                    "team_type": "PH-to-PH",
                    "status": "skipped",
                    "reason": f"No valid PH receiver found for sender high {sender_high} (candle {sender_idx})",
                    "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
                })
                i += 1

        # Process PL-to-PL trendlines
        i = 0
        while i < len(sorted_pl) - 1:
            sender_idx, sender_data = sorted_pl[i]
            sender_low = float(sender_data["low"])
            if i + 1 < len(sorted_pl):
                next_idx, next_data = sorted_pl[i + 1]
                next_low = float(next_data["low"])
                if next_low < sender_low:
                    trendline_log.append({
                        "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                        "symbol": symbol,
                        "timeframe": timeframe_str,
                        "team_type": "PL-to-PL",
                        "status": "skipped",
                        "reason": f"Immediate intruder PL found at candle {next_idx} (low {next_low}) lower than sender low {sender_low} (candle {sender_idx}), setting intruder as new sender",
                        "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
                    })
                    i += 1
                    continue
            best_receiver_idx = None
            best_receiver_low = float('inf')
            j = i + 1
            while j < len(sorted_pl):
                candidate_idx, candidate_data = sorted_pl[j]
                candidate_low = float(candidate_data["low"])
                if sender_low < candidate_low < best_receiver_low:
                    best_receiver_idx = candidate_idx
                    best_receiver_low = candidate_low
                j += 1
            if best_receiver_idx is not None:
                intruder_idx = find_intruder(sender_idx, best_receiver_idx, sender_low, is_ph=False)
                if intruder_idx is not None:
                    intruder_low = candle_bounds[intruder_idx]["low"]
                    trendline_log.append({
                        "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                        "symbol": symbol,
                        "timeframe": timeframe_str,
                        "team_type": "PL-to-PL",
                        "status": "skipped",
                        "reason": f"Intruder candle found at candle {intruder_idx} (low {intruder_low}) lower than sender low {sender_low} (candle {sender_idx}) between sender and receiver (candle {best_receiver_idx}), setting receiver as new sender",
                        "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
                    })
                    i = next((k for k, (idx, _) in enumerate(sorted_pl) if idx == best_receiver_idx), i + 1)
                    continue
                sender_pos = contour_positions.get(sender_idx)
                receiver_pos = contour_positions.get(best_receiver_idx)
                if sender_pos and receiver_pos:
                    sender_x = sender_pos["x"]
                    sender_y = sender_pos["y"] + sender_pos["height"]
                    receiver_x = receiver_pos["x"]
                    receiver_y = receiver_pos["y"] + receiver_pos["height"]
                    intersectors, slope = find_intersectors(sender_idx, best_receiver_idx, sender_x, sender_y, is_ph=False, receiver_price=best_receiver_low)
                    team_data = {
                        "sender": {"candle_number": sender_idx, "low": sender_low, "x": sender_x, "y": sender_y},
                        "receiver": {"candle_number": best_receiver_idx, "low": best_receiver_low, "x": receiver_x, "y": receiver_y},
                        "intersectors": [],
                        "trendlines": []
                    }
                    reason = ""
                    selected_idx = best_receiver_idx
                    selected_x = receiver_x
                    selected_y = receiver_y
                    selected_low = best_receiver_low
                    selected_type = "receiver"
                    selected_is_first = False
                    selected_marker = None
                    max_low = best_receiver_low
                    for idx, x, y, price, is_first, high_y, low_y, is_trendbreaker in intersectors:
                        marker = "star" if is_first else "circle"
                        team_data["intersectors"].append({
                            "candle_number": idx,
                            "low": price if is_first else None,
                            "x": x,
                            "y": low_y,
                            "is_first": is_first,
                            "is_trendbreaker": is_trendbreaker,
                            "marker": marker
                        })
                        if is_trendbreaker:
                            reason += f"; detected {'first' if is_first else 'subsequent'} intersector (candle {idx}, low {price if is_first else candle_bounds[idx]['low']}, x={x}, y={low_y}, marker={marker}, trendbreaker=True), skipped trendline generation"
                        else:
                            current_low = price if is_first else candle_bounds[idx]["low"]
                            if current_low > max_low:
                                max_low = current_low
                                selected_idx = idx
                                selected_x = x
                                selected_y = low_y
                                selected_low = current_low
                                selected_type = "intersector"
                                selected_is_first = is_first
                                selected_marker = "star" if is_first else "circle"
                            reason += f"; detected {'first' if is_first else 'subsequent'} intersector (candle {idx}, low {price if is_first else candle_bounds[idx]['low']}, x={x}, y={low_y}, marker={marker}, trendbreaker=False)"
                    # Draw trendline and check breakout only for the selected trendline
                    if selected_type == "receiver" and not intersectors:
                        # Find first breakout candle
                        first_breakout_idx = None
                        for check_idx in range(selected_idx - 1, -1, -1):
                            if check_idx in candle_bounds:
                                next_candle = candle_bounds[check_idx]
                                if (next_candle["high"] < candle_bounds[selected_idx]["high"] and
                                    next_candle["low"] < candle_bounds[selected_idx]["low"]):
                                    first_breakout_idx = check_idx
                                    break
                        # Extend trendline to first breakout candle if found
                        end_x = selected_x
                        end_y = selected_y
                        if first_breakout_idx is not None and first_breakout_idx in candle_bounds:
                            end_x = candle_bounds[first_breakout_idx]["x_left"]
                            if slope != float('inf'):
                                end_y = int(sender_y + slope * (end_x - sender_x))
                            else:
                                end_y = sender_y
                            reason += f"; extended trendline to x-axis of first breakout candle {first_breakout_idx} (x={end_x}, y={end_y})"
                        cv2.line(img, (sender_x, sender_y), (end_x, end_y), color=(0, 255, 255), thickness=1)
                        team_data["trendlines"].append({
                            "type": "receiver",
                            "candle_number": selected_idx,
                            "low": selected_low,
                            "x": end_x,
                            "y": end_y
                        })
                        pl_additional_trendlines.append({
                            "type": "receiver",
                            "candle_number": selected_idx,
                            "low": selected_low,
                            "x": end_x,
                            "y": end_y
                        })
                        reason = f"Drew PL-to-PL trendline from sender (candle {sender_idx}, low {sender_low}, x={sender_x}, y={sender_y}) to receiver (candle {selected_idx}, low={selected_low}, x={end_x}, y={end_y}) as no intersectors found" + reason
                        # Check for breakout candle for chevron arrow
                        if first_breakout_idx is not None:
                            start_idx = max(0, first_breakout_idx - minbreakoutcandleposition)
                            for check_idx in range(start_idx, -1, -1):
                                if check_idx in candle_bounds:
                                    next_candle = candle_bounds[check_idx]
                                    if (next_candle["high"] < candle_bounds[selected_idx]["high"] and
                                        next_candle["low"] < candle_bounds[selected_idx]["low"]):
                                        target_idx = check_idx
                                        target_x = candle_bounds[target_idx]["x_left"]
                                        target_y = candle_bounds[target_idx]["low_y"] - 10  # Add offset for PL
                                        draw_chevron_arrow(img, target_x + (candle_bounds[target_idx]["x_right"] - target_x) // 2, target_y, 'down', color=(128, 0, 128))
                                        reason += f"; for selected trendline to candle {selected_idx}, found first breakout candle {first_breakout_idx} (high={candle_bounds[first_breakout_idx]['high']}, low={candle_bounds[first_breakout_idx]['low']}), skipped {minbreakoutcandleposition} candles, selected target candle {target_idx} (low_y={target_y}, high={candle_bounds[target_idx]['high']}, low={candle_bounds[target_idx]['low']}) for purple chevron arrow at center x-axis with offset from low"
                                        break
                    elif selected_type == "intersector":
                        # Find first breakout candle
                        first_breakout_idx = None
                        for check_idx in range(selected_idx - 1, -1, -1):
                            if check_idx in candle_bounds:
                                next_candle = candle_bounds[check_idx]
                                if (next_candle["high"] < candle_bounds[selected_idx]["high"] and
                                    next_candle["low"] < candle_bounds[selected_idx]["low"]):
                                    first_breakout_idx = check_idx
                                    break
                        # Extend trendline to first breakout candle if found
                        end_x = selected_x
                        end_y = selected_y
                        if first_breakout_idx is not None and first_breakout_idx in candle_bounds:
                            end_x = candle_bounds[first_breakout_idx]["x_left"]
                            if slope != float('inf'):
                                end_y = int(sender_y + slope * (end_x - sender_x))
                            else:
                                end_y = sender_y
                            reason += f"; extended trendline to x-axis of first breakout candle {first_breakout_idx} (x={end_x}, y={end_y})"
                        cv2.line(img, (sender_x, sender_y), (end_x, end_y), color=(0, 255, 255), thickness=1)
                        if selected_is_first:
                            star_points = [
                                [selected_x, selected_y - 15], [selected_x + 4, selected_y - 5], [selected_x + 14, selected_y - 5],
                                [selected_x + 5, selected_y + 2], [selected_x + 10, selected_y + 12], [selected_x, selected_y + 7],
                                [selected_x - 10, selected_y + 12], [selected_x - 5, selected_y + 2], [selected_x - 14, selected_y - 5],
                                [selected_x - 4, selected_y - 5]
                            ]
                            cv2.fillPoly(img, [np.array(star_points)], color=(0, 255, 255))
                        else:
                            cv2.circle(img, (selected_x, selected_y), 5, color=(0, 255, 255), thickness=-1)
                        team_data["trendlines"].append({
                            "type": "intersector",
                            "candle_number": selected_idx,
                            "low": selected_low,
                            "x": end_x,
                            "y": end_y,
                            "is_first": selected_is_first,
                            "is_trendbreaker": False,
                            "marker": selected_marker
                        })
                        pl_additional_trendlines.append({
                            "type": "intersector",
                            "candle_number": selected_idx,
                            "low": selected_low,
                            "x": end_x,
                            "y": end_y,
                            "is_first": selected_is_first,
                            "is_trendbreaker": False,
                            "marker": selected_marker
                        })
                        reason = f"Drew PL-to-PL trendline from sender (candle {sender_idx}, low {sender_low}, x={sender_x}, y={sender_y}) to intersector (candle {selected_idx}, low={selected_low}, x={end_x}, y={end_y}, marker={selected_marker}, trendbreaker=False) with highest low" + reason
                        # Check for breakout candle for chevron arrow
                        if first_breakout_idx is not None:
                            start_idx = max(0, first_breakout_idx - minbreakoutcandleposition)
                            for check_idx in range(start_idx, -1, -1):
                                if check_idx in candle_bounds:
                                    next_candle = candle_bounds[check_idx]
                                    if (next_candle["high"] < candle_bounds[selected_idx]["high"] and
                                        next_candle["low"] < candle_bounds[selected_idx]["low"]):
                                        target_idx = check_idx
                                        target_x = candle_bounds[target_idx]["x_left"]
                                        target_y = candle_bounds[target_idx]["low_y"] - 10  # Add offset for PL
                                        draw_chevron_arrow(img, target_x + (candle_bounds[target_idx]["x_right"] - target_x) // 2, target_y, 'down', color=(128, 0, 128))
                                        reason += f"; for selected trendline to candle {selected_idx}, found first breakout candle {first_breakout_idx} (high={candle_bounds[first_breakout_idx]['high']}, low={candle_bounds[first_breakout_idx]['low']}), skipped {minbreakoutcandleposition} candles, selected target candle {target_idx} (low_y={target_y}, high={candle_bounds[target_idx]['high']}, low={candle_bounds[target_idx]['low']}) for purple chevron arrow at center x-axis with offset from low"
                                        break
                    else:
                        reason = f"No PL-to-PL trendline drawn from sender (candle {sender_idx}, low {sender_low}, x={sender_x}, y={sender_y}) as first intersector is trendbreaker" + reason
                    trendline_log.append({
                        "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                        "symbol": symbol,
                        "timeframe": timeframe_str,
                        "team_type": "PL-to-PL",
                        "status": "success" if team_data["trendlines"] else "skipped",
                        "reason": reason,
                        "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
                    })
                    pl_teams.append(team_data)
                    i = next((k for k, (idx, _) in enumerate(sorted_pl) if idx == best_receiver_idx), i + 1) + 1
                else:
                    error_log.append({
                        "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                        "error": f"Missing contour positions for PL sender (candle {sender_idx}) or receiver (candle {best_receiver_idx})",
                        "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
                    })
                    i += 1
            else:
                trendline_log.append({
                    "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                    "symbol": symbol,
                    "timeframe": timeframe_str,
                    "team_type": "PL-to-PL",
                    "status": "skipped",
                    "reason": f"No valid PL receiver found for sender low {sender_low} (candle {sender_idx})",
                    "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
                })
                i += 1

        # Save the image
        cv2.imwrite(output_image_path, img)
        log_and_print(
            f"Chart with colored contours (dim green for up, red for down), "
            f"PH/PL markers (blue for PH, purple for PL), "
            f"single trendlines (blue for PH-to-PH to lowest high intersector or receiver extended to first breakout candle, yellow for PL-to-PL to highest low intersector or receiver extended to first breakout candle), "
            f"intersector markers (blue star/circle for PH selected intersector, yellow star/circle for PL selected intersector), "
            f"blue upward chevron arrow on first PH breakout candle after skipping {minbreakoutcandleposition} candles from initial breakout with higher low and higher high for last PH trendline at center x-axis with offset, "
            f"purple downward chevron arrow on first PL breakout candle after skipping {minbreakoutcandleposition} candles from initial breakout with lower high and lower low for last PL trendline at center x-axis with offset), "
            f"saved for {symbol} ({timeframe_str}) at {output_image_path}",
            "SUCCESS"
        )

        # Save contour data and trendline log
        contour_data = {
            "total_count": total_count,
            "green_candle_count": green_count,
            "red_candle_count": red_count,
            "candle_contours": [],
            "ph_teams": ph_teams,
            "pl_teams": pl_teams,
            "ph_additional_trendlines": ph_additional_trendlines,
            "pl_additional_trendlines": pl_additional_trendlines
        }
        for i, contour in enumerate(all_contours):
            x, y, w, h = cv2.boundingRect(contour)
            candle_type = "green" if green_mask[y + h // 2, x + w // 2] > 0 else "red" if red_mask[y + h // 2, x + w // 2] > 0 else "unknown"
            contour_data["candle_contours"].append({
                "candle_number": i,
                "type": candle_type,
                "x": x + w // 2,
                "y": y,
                "width": w,
                "height": h,
                "is_ph": i in ph_indices,
                "is_pl": i in pl_indices
            })
        try:
            with open(contour_json_path, 'w') as f:
                json.dump(contour_data, f, indent=4)
            log_and_print(
                f"Contour count and trendline data saved for {symbol} ({timeframe_str}) at {contour_json_path} "
                f"with total_count={total_count} (green={green_count}, red={red_count}, PH={len(ph_indices)}, PL={len(pl_indices)}, "
                f"PH_teams={len(ph_teams)}, PL_teams={len(pl_teams)}, "
                f"PH_trendlines={len(ph_additional_trendlines)}, PL_trendlines={len(pl_additional_trendlines)})",
                "SUCCESS"
            )
        except Exception as e:
            error_log.append({
                "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                "error": f"Failed to save contour count for {symbol} ({timeframe_str}): {str(e)}",
                "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
            })
            save_errors(error_log)
            log_and_print(f"Failed to save contour count for {symbol} ({timeframe_str}): {str(e)}", "ERROR")
        try:
            trendline_log.insert(0, {
                "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                "symbol": symbol,
                "timeframe": timeframe_str,
                "team_type": "initial",
                "status": "info",
                "reason": f"Found {len(ph_indices)} PH points and {len(pl_indices)} PL points",
                "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
            })
            with open(trendline_log_json_path, 'w') as f:
                json.dump(trendline_log, f, indent=4)
            log_and_print(f"Trendline log saved for {symbol} ({timeframe_str}) at {trendline_log_json_path}", "SUCCESS")
        except Exception as e:
            error_log.append({
                "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                "error": f"Failed to save trendline log for {symbol} ({timeframe_str}): {str(e)}",
                "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
            })
            save_errors(error_log)
            log_and_print(f"Failed to save trendline log for {symbol} ({timeframe_str}): {str(e)}", "ERROR")
        log_and_print(
            f"Detected {total_count} candle contours for {symbol} ({timeframe_str}) (green={green_count}, red={red_count}), "
            f"counted from newest to oldest, with {len(ph_indices)} PH and {len(pl_indices)} PL marked, "
            f"{len(ph_teams)} PH-to-PH teams, {len(pl_teams)} PL-to-PL teams, "
            f"{len(ph_additional_trendlines)} PH-to-PH trendlines (extended to first breakout candle), "
            f"{len(pl_additional_trendlines)} PL-to-PL trendlines (extended to first breakout candle)",
            "INFO"
        )
    except Exception as e:
        error_log.append({
            "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
            "error": f"Failed to detect contours or draw trendlines for {symbol} ({timeframe_str}) in chart.png: {str(e)}",
            "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
        })
        save_errors(error_log)
        log_and_print(f"Failed to detect contours or draw trendlines for {symbol} ({timeframe_str}) in chart.png: {str(e)}", "ERROR")
    return error_log

    
def crop_chart(chart_path, symbol, timeframe_str, timeframe_folder):
    """Crop the saved chart.png and chartanalysed.png images, then detect candle contours only for chart.png."""
    error_log = []
    chart_analysed_path = os.path.join(timeframe_folder, "chartanalysed.png")

    try:
        # Crop chart.png
        with Image.open(chart_path) as img:
            right = 8
            left = 80
            top = 80
            bottom = 70
            crop_box = (left, top, img.width - right, img.height - bottom)
            cropped_img = img.crop(crop_box)
            cropped_img.save(chart_path, "PNG")
            log_and_print(f"Chart cropped for {symbol} ({timeframe_str}) at {chart_path}", "SUCCESS")

        # Detect contours for chart.png only
        contour_errors = detect_candle_contours(chart_path, symbol, timeframe_str, timeframe_folder)
        error_log.extend(contour_errors)

        # Crop chartanalysed.png if it exists
        if os.path.exists(chart_analysed_path):
            with Image.open(chart_analysed_path) as img:
                crop_box = (left, top, img.width - right, img.height - bottom)
                cropped_img = img.crop(crop_box)
                cropped_img.save(chart_analysed_path, "PNG")
                log_and_print(f"Analysed chart cropped for {symbol} ({timeframe_str}) at {chart_analysed_path}", "SUCCESS")
        else:
            error_log.append({
                "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
                "error": f"chartanalysed.png not found for {symbol} ({timeframe_str})",
                "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
            })
            log_and_print(f"chartanalysed.png not found for {symbol} ({timeframe_str})", "WARNING")

    except Exception as e:
        error_log.append({
            "timestamp": datetime.now(pytz.timezone('Africa/Lagos')).strftime('%Y-%m-%d %H:%M:%S.%f+01:00'),
            "error": f"Failed to crop charts for {symbol} ({timeframe_str}): {str(e)}",
            "broker": mt5.terminal_info().name if mt5.terminal_info() else "unknown"
        })
        save_errors(error_log)
        log_and_print(f"Failed to crop charts for {symbol} ({timeframe_str}): {str(e)}", "ERROR")

    return error_log
  

def fetch_charts_all_brokers(
    bars,
    neighborcandles_left,
    neighborcandles_right
):
    """Main function to fetch OHLCV data, save charts, crop them, apply arrow detection, and save candle details."""
    error_log = []
    log_and_print("Starting chart generation process for all brokers with their respective symbols", "INFO")

    # Clear chart folders for all brokers
    for broker_name, config in brokersdictionary.items():
        log_and_print(f"Clearing chart folder for broker: {broker_name}", "INFO")
        success_clear, clear_errors = clear_chart_folder(config["BASE_FOLDER"])
        error_log.extend(clear_errors)
        if not success_clear:
            log_and_print(f"Failed to clear chart folder for {broker_name}, continuing", "ERROR")

    # Get symbols for each broker
    broker_symbols = {}
    for broker_name, config in brokersdictionary.items():
        log_and_print(f"Initializing MT5 for broker: {broker_name}", "INFO")
        success, init_errors = initialize_mt5(
            config["TERMINAL_PATH"],
            config["LOGIN_ID"],
            config["PASSWORD"],
            config["SERVER"]
        )
        error_log.extend(init_errors)
        if not success:
            log_and_print(f"MT5 initialization failed for {broker_name}, skipping", "ERROR")
            mt5.shutdown()
            continue

        symbols, sym_errors = get_symbols()
        error_log.extend(sym_errors)
        broker_symbols[broker_name] = sorted(list(symbols))
        mt5.shutdown()
        log_and_print(f"Retrieved {len(symbols)} symbols for {broker_name}", "INFO")

    if not broker_symbols:
        log_and_print("No symbols retrieved from any broker, aborting", "ERROR")
        save_errors(error_log)
        return False

    # Initialize remaining symbols for each broker
    remaining_symbols = {broker: symbols.copy() for broker, symbols in broker_symbols.items()}
    brokers = list(broker_symbols.keys())
    broker_indices = {broker: 0 for broker in brokers}

    # Process symbols in a round-robin fashion
    while any(remaining_symbols[broker] for broker in brokers):
        for broker_name in brokers:
            if not remaining_symbols[broker_name]:
                continue

            log_and_print(f"{broker_name}", "INFO")
            log_and_print("")
            log_and_print(f"Processing next symbol for broker: {broker_name}", "INFO")
            current_index = broker_indices[broker_name]
            if current_index >= len(remaining_symbols[broker_name]):
                continue
            symbol = remaining_symbols[broker_name][current_index]

            config = brokersdictionary[broker_name]
            success, init_errors = initialize_mt5(
                config["TERMINAL_PATH"],
                config["LOGIN_ID"],
                config["PASSWORD"],
                config["SERVER"]
            )
            error_log.extend(init_errors)
            if not success:
                log_and_print(f"MT5 initialization failed for {broker_name} while processing {symbol}, skipping", "ERROR")
                mt5.shutdown()
                continue

            log_and_print(f"Processing symbol {symbol} for broker: {broker_name}", "INFO")
            symbol_folder = os.path.join(config["BASE_FOLDER"], symbol.replace(' ', '_'))
            os.makedirs(symbol_folder, exist_ok=True)

            for timeframe_str, mt5_timeframe in TIMEFRAME_MAP.items():
                log_and_print(f"Processing timeframe: {timeframe_str} for {symbol}", "INFO")
                timeframe_folder = os.path.join(symbol_folder, timeframe_str)
                os.makedirs(timeframe_folder, exist_ok=True)

                df, data_errors = fetch_ohlcv_data(symbol, mt5_timeframe, bars)
                error_log.extend(data_errors)
                if df is None:
                    continue

                df['symbol'] = symbol
                # Generate chart and get PH/PL labels
                chart_path, chart_errors, ph_labels, pl_labels = generate_and_save_chart(
                    df, symbol, timeframe_str, timeframe_folder,
                    neighborcandles_left, neighborcandles_right
                )
                error_log.extend(chart_errors)

                # Save candle data with PH/PL labels
                candle_errors = save_candle_data(df, symbol, timeframe_str, timeframe_folder, ph_labels, pl_labels)
                error_log.extend(candle_errors)

                if chart_path:
                    crop_errors = crop_chart(chart_path, symbol, timeframe_str, timeframe_folder)
                    error_log.extend(crop_errors)
                log_and_print("", "INFO")

            mt5.shutdown()
            log_and_print("", "INFO")
            broker_indices[broker_name] += 1
            if broker_indices[broker_name] < len(remaining_symbols[broker_name]):
                remaining_symbols[broker_name][broker_indices[broker_name]]
            else:
                remaining_symbols[broker_name] = []

    save_errors(error_log)
    log_and_print("Chart generation process completed for all brokers with their respective symbols", "SUCCESS")
    return len(error_log) == 0

if __name__ == "__main__":
    success = fetch_charts_all_brokers(
        bars=251,
        neighborcandles_left=10,
        neighborcandles_right=15
    )
    if success:
        log_and_print("Chart generation, cropping, arrow detection, PH/PL analysis, and candle data saving completed successfully for all brokers!", "SUCCESS")
    else:
        log_and_print("Process failed. Check error log for details.", "ERROR")
        