"""
Methodology:
1. Download the pool data
cryo logs \
    --label uniswap_v3_swaps \
    --blocks 20_020_000: \
    --reorg-buffer 1000 \
    --subdirs datatype \
    --contract 0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640 \
    --event-signature "Swap(address indexed sender, address indexed recipient, int256 amount0, int256 amount1, uint160 sqrtPriceX96, uint128 liquidity, int24 tick)" \
    --topic0 0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67 \
    --rpc <RPC_URL>



2. Load the pool data
3. Calculate:
    - swap price from event__sqrtPriceX96_f64
    - swap fee in input token
4. Download ETH price data
5. Normalize time references between the two datasets
6. Calculate:
    - swap price in USD
    - swap fee in USD
    - price difference between swap price and oracle price (%)
    - some stats about price difference (average...)
    - total fees in USD
7. Calculate dynamic fee curve with different c values
8. Calculate total fees in USD for each dynamic fee curve and choose the one that maximizes the total fees
"""

import glob
import time
import requests
import numpy as np
import pandas as pd
from web3 import Web3
from colorama import Fore
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.special import expit


print("Starting...")
start_time = time.time()
print()


RPC_URL = "<RPC_URL>"
web3 = Web3(Web3.HTTPProvider(RPC_URL))

###################################################### CONSTANTS ######################################################

SWAP_FEE = 0.0005  # 0.05% swap fee
WETH_DECIMALS = 18
USDC_DECIMALS = 6

CHAIN_DATA_FILE_PATH = "<CHAIN_DATA_FILE_PATH>"
OUTPUT_FILE_PATH = "<OUTPUT_FILE_PATH>"
MAX_DATAPOINTS = 500_000
NEZLOBIN_C_VALUE = 0.9

# Sigmoid S-Curve Parameters
C0 = 0
C1 = 1
C2 = 600
C3 = 0.01

SHOW_PLOTS = True

###################################################### HELPERS ######################################################


def sqrtPriceX96_to_price(sqrtPriceX96: int) -> float:
    Q96 = 2**96
    return 1 / ((sqrtPriceX96 / Q96) ** 2) * 10 ** (WETH_DECIMALS - USDC_DECIMALS)


def fetch_price_data(start_time: int, end_time: int, symbol: str = "ETHUSDC", limit=1000):
    start_time_ms = start_time * 1000
    end_time_ms = end_time * 1000
    url = f"https://api.binance.com/api/v3/aggTrades?symbol={symbol}&startTime={start_time_ms}&endTime={end_time_ms}&limit={limit}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    print(Fore.RED + f"Error fetching data: {response.status_code}, {response.text}")
    return []


# Function to find the closest timestamp price
def find_closest_price(timestamp: int) -> float:
    # Find the closest timestamp in all_prices_df
    timestamp_ms = timestamp * 1000
    closest_idx = (all_prices_df["T"] - timestamp_ms).abs().idxmin()
    return all_prices_df.loc[closest_idx, "p"]


def calculate_swap_fee(row: pd.Series) -> float:
    amount0 = int(row["event__amount0_string"])
    amount1 = int(row["event__amount1_string"])
    if amount0 < 0:
        return abs(SWAP_FEE * amount0 * 10**-USDC_DECIMALS)
    return abs(SWAP_FEE * amount1 * 10**-WETH_DECIMALS)


def convert_swap_fee_to_usd(row: pd.Series) -> float:
    amount0 = int(row["event__amount0_string"])
    swap_fee = row["swap_fee"]
    price_usd_float = float(row["price_usd"].replace(",", ""))
    if amount0 < 0:
        return swap_fee  # Assuming 1 USD = 1 USDC here
    return swap_fee * price_usd_float


def calculate_nezlobin_fee(row: pd.Series) -> float:
    amount0 = int(row["event__amount0_string"])
    amount1 = int(row["event__amount1_string"])
    if amount0 < 0:
        return abs(row["dynamic_nezlobin_fee_percentage"] * amount0 * 10**-USDC_DECIMALS)
    return abs(row["dynamic_nezlobin_fee_percentage"] * amount1 * 10**-WETH_DECIMALS)


def convert_nezlobin_fee_to_usd(row: pd.Series) -> float:
    amount0 = int(row["event__amount0_string"])
    swap_fee = row["swap_fee_nezlobin"]
    price_usd_float = float(row["price_usd"].replace(",", ""))
    if amount0 < 0:
        return swap_fee  # Assuming 1 USD = 1 USDC here
    return swap_fee * price_usd_float


def calculate_sigmund_fee(row: pd.Series) -> float:
    amount0 = int(row["event__amount0_string"])
    amount1 = int(row["event__amount1_string"])
    if amount0 < 0:
        return abs(row["dynamic_sigmund_fee_percentage"] * amount0 * 10**-USDC_DECIMALS)
    return abs(row["dynamic_sigmund_fee_percentage"] * amount1 * 10**-WETH_DECIMALS)


def convert_sigmund_fee_to_usd(row: pd.Series) -> float:
    amount0 = int(row["event__amount0_string"])
    swap_fee = row["swap_fee_sigmund"]
    price_usd_float = float(row["price_usd"].replace(",", ""))
    if amount0 < 0:
        return swap_fee  # Assuming 1 USD = 1 USDC here
    return swap_fee * price_usd_float


# Initialize a dictionary to cache fetched block timestamps
fetched_timestamps = {}


# Function to fetch block timestamp
def fetch_block_timestamp(block_number: int) -> int:
    global fetched_blocks
    if block_number not in fetched_timestamps:  # Check if timestamp is already fetched
        print(Fore.YELLOW + f"Fetching block {fetched_blocks}/{new_blocks}...")
        try:
            # Fetch block data using web3
            block = web3.eth.get_block(block_number)
            fetched_timestamps[block_number] = block["timestamp"]
        except Exception as e:
            print(Fore.RED + f"Error fetching block {block_number}: {str(e)}")
            fetched_timestamps[block_number] = None  # Handle error by setting None
            time.sleep(0.1)  # Add a small delay to avoid rate limits
        fetched_blocks += 1
    return fetched_timestamps[block_number]


def get_binance_prices_df(new_rows: pd.DataFrame) -> pd.DataFrame:
    # Fetch Binance prices for new rows
    all_prices_df = pd.DataFrame()
    min_timestamp = new_rows["timestamp"].min()
    max_timestamp = new_rows["timestamp"].max()

    chunk_size = 60  # 1 minute in seconds
    current_start = min_timestamp

    min_timestamp = new_rows["timestamp"].min()
    max_timestamp = new_rows["timestamp"].max()

    chunk_size = 60000  # 1 minute in ms
    current_start = min_timestamp

    # Calculate total chunks for progress tracking
    total_chunks = ((max_timestamp - min_timestamp) // chunk_size) + 1

    for i, _ in enumerate(range(min_timestamp, max_timestamp, chunk_size), start=1):
        current_end = min(current_start + chunk_size, max_timestamp)

        # Print the progress indicator
        print(Fore.YELLOW, f"Fetching Binance prices chunk {i}/{total_chunks}...")

        # Fetch the price data
        price_data = fetch_price_data(start_time=current_start, end_time=current_end)

        if price_data:
            price_df = pd.DataFrame(price_data)
            price_df["T"] = price_df["T"].astype(int)
            price_df["p"] = price_df["p"].str.replace(",", "").astype(float)
            all_prices_df = pd.concat([all_prices_df, price_df], ignore_index=True)

        current_start = current_end + 1
        time.sleep(0.2)  # Adjust sleep as needed to stay within rate limits
    return all_prices_df


def calculate_abs_sig_fee_percentage(price_delta: float) -> float:
    exponent = -C2 * (price_delta - C3)
    # fee = C0 + C1 / (1 + np.exp(exponent))
    fee = C0 + C1 * expit(exponent)
    return fee


###################################################### MAIN ######################################################


# Load existing processed data if available
try:
    processed_df = pd.read_csv(OUTPUT_FILE_PATH)
    print(Fore.GREEN + f"Loaded {len(processed_df)} rows from {OUTPUT_FILE_PATH}.")
except FileNotFoundError:
    processed_df = pd.DataFrame()
    print(Fore.GREEN + "No existing data found, starting fresh.")


# Load all parquet files and combine them
parquet_files = glob.glob(CHAIN_DATA_FILE_PATH + "*.parquet")
dataframes = [pd.read_parquet(file) for file in parquet_files]
combined_df = pd.concat(dataframes, ignore_index=True)
df = combined_df


# Filter the last n entries for testing purposes
# df = df.tail(MAX_DATAPOINTS)  # <-- Uncomment this line for testing and comment it back after testing


columns_to_remove = [
    "event__liquidity_string",
    "transaction_hash",
    "address",
    "topic0",
    "n_data_bytes",
    "chain_id",
    "event__sender",
    "event__recipient",
    "event__amount0_binary",
    "event__amount0_f64",
    "event__amount1_binary",
    "event__amount1_f64",
    "event__sqrtPriceX96_binary",
    "event__sqrtPriceX96_f64",
    "event__liquidity_binary",
    "event__liquidity_string",
]

df = df.drop(columns=columns_to_remove)

# Identify new rows to process
if not processed_df.empty:
    new_rows = df[~df["block_number"].isin(processed_df["block_number"])]
else:
    new_rows = df.copy()

# Process only new rows
if not new_rows.empty:
    print(Fore.GREEN + f"Processing {len(new_rows)} new rows...")

    new_rows["price ETH/USD"] = new_rows.apply(
        lambda row: sqrtPriceX96_to_price(int(row["event__sqrtPriceX96_string"])), axis=1
    )
    new_rows["price ETH/USD"] = new_rows["price ETH/USD"].apply(lambda x: f"{x:,.2f}")

    # Fetch block timestamps for new rows
    new_blocks = new_rows["block_number"].nunique()
    fetched_blocks = 1
    new_rows["timestamp"] = new_rows["block_number"].apply(fetch_block_timestamp)
    new_rows["timestamp"] = new_rows["timestamp"].interpolate(method="linear").astype("Int64")

    all_prices_df = get_binance_prices_df(new_rows)
    # Sort and reset index after fetching all chunks
    all_prices_df.sort_values("T", inplace=True)
    all_prices_df.reset_index(drop=True, inplace=True)

    new_rows["price_usd"] = new_rows["timestamp"].apply(find_closest_price)
    new_rows["price_usd"] = new_rows["price_usd"].apply(lambda x: f"{x:,.2f}")

    # Calculate additional columns
    new_rows["price_delta_percentage"] = (
        (
            new_rows["price_usd"].str.replace(",", "").astype(float)
            - new_rows["price ETH/USD"].str.replace(",", "").astype(float)
        )
        / new_rows["price_usd"].str.replace(",", "").astype(float)
        * 100
    )

    new_rows["swap_fee"] = new_rows.apply(calculate_swap_fee, axis=1)
    new_rows["swap_fee_usd"] = new_rows.apply(convert_swap_fee_to_usd, axis=1)

    # Combine new and processed data
    processed_df = pd.concat([processed_df, new_rows], ignore_index=True)

    # Sort by block number
    processed_df = processed_df.sort_values(by="block_number", ascending=True)

    # Save combined data to file
    processed_df.to_csv(OUTPUT_FILE_PATH, index=False)
    print(Fore.GREEN + f"Saved {len(processed_df)} rows to {OUTPUT_FILE_PATH}.")

    processed_df["price ETH/USD"] = processed_df.apply(
        lambda row: sqrtPriceX96_to_price(int(row["event__sqrtPriceX96_string"])), axis=1
    )

    processed_df["price ETH/USD"] = processed_df["price ETH/USD"].apply(lambda x: f"{x:,.2f}")
    processed_df["price ETH/USD"] = pd.to_numeric(processed_df["price ETH/USD"].str.replace(",", ""), errors="coerce")

    # Calculate the percentage difference for the 'price ETH/USD' column

    processed_df["pool_price_delta_perc"] = processed_df["price ETH/USD"].pct_change() * 100

    # ########## DYNAMIC NEZLOBIN FEE ##########

    # TODO:
    # 1 - Check Nezlobin Fee Calculation
    #
    # 2  -Add Logic for Fee = 0 if nezlobin fee > swap fee AND swap usd amount > 1_000_000 and price_delta < nezlobin_fee_usd + gas_fee_usd (calculate assuming average ct gas fee for swap) - swap not executed

    # Set the first row value to 0.05% (or 0.0005 in decimal form)
    processed_df.loc[0, "dynamic_nezlobin_fee_percentage"] = SWAP_FEE

    last_computed_fee = SWAP_FEE

    for i in range(1, len(processed_df)):
        # Check if it's the first row with a given block number (TOP OF THE BLOCK)
        if processed_df.loc[i, "block_number"] != processed_df.loc[i - 1, "block_number"]:
            # Convert to float to ensure proper comparison
            price_delta = float(processed_df.loc[i, "pool_price_delta_perc"])
            amount1 = float(processed_df.loc[i, "event__amount1_string"])

            # Calculate the fee for the first occurrence of the block
            if price_delta < 0:  # AMM_PRICE > BINANCE PRICE -> ARBITRAGE DIRECTION: SELL WETH
                if amount1 < 0:  # SELL WETH
                    # last_computed_fee = last_computed_fee * (
                    #     1 + NEZLOBIN_C_VALUE * abs(processed_df.loc[i, "pool_price_delta_perc"])
                    # )  # FEE INCREASE
                    last_computed_fee = SWAP_FEE * (1 + NEZLOBIN_C_VALUE)
                else:
                    # last_computed_fee = last_computed_fee * (
                    #     1 - NEZLOBIN_C_VALUE * abs(processed_df.loc[i, "pool_price_delta_perc"])
                    # )  # FEE DECREASE
                    last_computed_fee = SWAP_FEE * (1 - NEZLOBIN_C_VALUE)
            elif price_delta > 0:  # AMM_PRICE < BINANCE PRICE -> ARBITRAGE DIRECTION: BUY WETH
                if amount1 > 0:  # BUY WETH
                    # last_computed_fee = last_computed_fee * (
                    #     1 + NEZLOBIN_C_VALUE * abs(processed_df.loc[i, "pool_price_delta_perc"])
                    # )  # FEE INCREASE
                    last_computed_fee = SWAP_FEE * (1 + NEZLOBIN_C_VALUE)
                else:
                    # last_computed_fee = last_computed_fee * (
                    #     1 - NEZLOBIN_C_VALUE * abs(processed_df.loc[i, "pool_price_delta_perc"])
                    # )  # FEE DECREASE
                    last_computed_fee = SWAP_FEE * (1 - NEZLOBIN_C_VALUE)

            # Assign the computed fee to the first row of the block
            processed_df.loc[i, "dynamic_nezlobin_fee_percentage"] = last_computed_fee
        else:
            # For all subsequent rows in the same block, use the already computed fee
            processed_df.loc[i, "dynamic_nezlobin_fee_percentage"] = last_computed_fee

    processed_df["swap_fee_nezlobin"] = processed_df.apply(calculate_nezlobin_fee, axis=1)
    processed_df["swap_fee_nezlobin_usd"] = processed_df.apply(convert_nezlobin_fee_to_usd, axis=1)

    processed_df.to_csv(OUTPUT_FILE_PATH, index=False)

    # ########## DYNAMIC SIGMOID S-CURVE FEE ##########

    # # Set the first row value to 0.05% (or 0.0005 in decimal form)
    processed_df.loc[0, "dynamic_sigmund_fee_percentage"] = SWAP_FEE

    last_computed_fee = SWAP_FEE

    for i in range(1, len(processed_df)):
        # Check if it's the first row with a given block number (TOP OF THE BLOCK)
        if processed_df.loc[i, "block_number"] != processed_df.loc[i - 1, "block_number"]:
            # Convert to float to ensure proper comparison
            price_delta = float(processed_df.loc[i, "price_delta_percentage"])
            abs_sig_fee_percentage = calculate_abs_sig_fee_percentage(abs(price_delta))
            amount1 = float(processed_df.loc[i, "event__amount1_string"])

            # Calculate the fee for the first occurrence of the block
            if price_delta < 0:  # AMM_PRICE > BINANCE PRICE -> ARBITRAGE DIRECTION: SELL WETH
                if amount1 < 0:  # SELL WETH
                    last_computed_fee = SWAP_FEE * (1 + abs_sig_fee_percentage)
                else:
                    last_computed_fee = SWAP_FEE * (1 - abs_sig_fee_percentage)
            elif price_delta > 0:  # AMM_PRICE < BINANCE PRICE -> ARBITRAGE DIRECTION: BUY WETH
                if amount1 > 0:  # BUY WETH
                    last_computed_fee = SWAP_FEE * (1 + abs_sig_fee_percentage)
                else:
                    last_computed_fee = SWAP_FEE * (1 - abs_sig_fee_percentage)

            # Assign the computed fee to the first row of the block
            processed_df.loc[i, "dynamic_sigmund_fee_percentage"] = last_computed_fee
        else:
            # For all subsequent rows in the same block, use the already computed fee
            processed_df.loc[i, "dynamic_sigmund_fee_percentage"] = last_computed_fee

    processed_df["swap_fee_sigmund"] = processed_df.apply(calculate_sigmund_fee, axis=1)
    processed_df["swap_fee_sigmund_usd"] = processed_df.apply(convert_sigmund_fee_to_usd, axis=1)

    processed_df.to_csv(OUTPUT_FILE_PATH, index=False)
else:

    print(Fore.GREEN + "No new rows to process.")


# ###################################################### TABLES ######################################################

# pd.set_option("display.float_format", "{:.2f}".format)
# pd.set_option("display.max_columns", 23)

pd.set_option("display.max_rows", 20)
print()
print(processed_df.head(100))
print()


# Perform data analysis


# Flat Fee
total_swap_fee_usd = processed_df["swap_fee_usd"].sum()
mean_swap_fee_usd = processed_df["swap_fee_usd"].mean()
median_swap_fee_usd = processed_df["swap_fee_usd"].median()
std_swap_fee_usd = processed_df["swap_fee_usd"].std()
min_swap_fee_usd = processed_df["swap_fee_usd"].min()
max_swap_fee_usd = processed_df["swap_fee_usd"].max()

flat_swap_fees_summary = {
    "Total Swap Fee USD": total_swap_fee_usd,
    "Median Swap Fee USD": median_swap_fee_usd,
    "Mean Swap Fee USD": mean_swap_fee_usd,
    "Std Dev Swap Fee USD": std_swap_fee_usd,
    "Min Swap Fee USD": min_swap_fee_usd,
    "Max Swap Fee USD": max_swap_fee_usd,
}


# Nezlobin Fee
total_swap_fee_nezlobin_usd = processed_df["swap_fee_nezlobin_usd"].sum()
mean_swap_fee_nezlobin_usd = processed_df["swap_fee_nezlobin_usd"].mean()
median_swap_fee_nezlobin_usd = processed_df["swap_fee_nezlobin_usd"].median()
std_swap_fee_nezlobin_usd = processed_df["swap_fee_nezlobin_usd"].std()
min_swap_fee_nezlobin_usd = processed_df["swap_fee_nezlobin_usd"].min()
max_swap_fee_nezlobin_usd = processed_df["swap_fee_nezlobin_usd"].max()

nezlobin_swap_fees_summary = {
    "Total Swap Fee USD": total_swap_fee_nezlobin_usd,
    "Median Swap Fee USD": median_swap_fee_nezlobin_usd,
    "Mean Swap Fee USD": mean_swap_fee_nezlobin_usd,
    "Std Dev Swap Fee USD": std_swap_fee_nezlobin_usd,
    "Min Swap Fee USD": min_swap_fee_nezlobin_usd,
    "Max Swap Fee USD": max_swap_fee_nezlobin_usd,
}

# Sigmund Fee
total_swap_fee_sigmund_usd = processed_df["swap_fee_sigmund_usd"].sum()
mean_swap_fee_sigmund_usd = processed_df["swap_fee_sigmund_usd"].mean()
median_swap_fee_sigmund_usd = processed_df["swap_fee_sigmund_usd"].median()
std_swap_fee_sigmund_usd = processed_df["swap_fee_sigmund_usd"].std()
min_swap_fee_sigmund_usd = processed_df["swap_fee_sigmund_usd"].min()
max_swap_fee_sigmund_usd = processed_df["swap_fee_sigmund_usd"].max()

sigmund_swap_fees_summary = {
    "Total Swap Fee USD": total_swap_fee_sigmund_usd,
    "Median Swap Fee USD": median_swap_fee_sigmund_usd,
    "Mean Swap Fee USD": mean_swap_fee_sigmund_usd,
    "Std Dev Swap Fee USD": std_swap_fee_sigmund_usd,
    "Min Swap Fee USD": min_swap_fee_sigmund_usd,
    "Max Swap Fee USD": max_swap_fee_sigmund_usd,
}

# Price Delta
average_price_delta_percentage = processed_df["price_delta_percentage"].abs().mean()
median_price_delta_percentage = processed_df["price_delta_percentage"].abs().median()
std_price_delta_percentage = processed_df["price_delta_percentage"].abs().std()
min_price_delta_percentage = processed_df["price_delta_percentage"].abs().min()
max_price_delta_percentage = processed_df["price_delta_percentage"].abs().max()

price_delta_sumary = {
    "Mean Price Delta Percentage": average_price_delta_percentage,
    "Median Price Delta Percentage": median_price_delta_percentage,
    "Std Dev Price Delta Percentage": std_price_delta_percentage,
    "Min Price Delta Percentage": min_price_delta_percentage,
    "Max Price Delta Percentage": max_price_delta_percentage,
}


# Print the summary to the shell
print()
print(Fore.GREEN + "Flat Swap Fees Analysis Summary:")
for metric, value in flat_swap_fees_summary.items():
    print(Fore.BLUE + f"{metric}: {value:.6f}")
print()

print()
print(Fore.GREEN + "Nezlobin Swap Fees Analysis Summary:")
for metric, value in nezlobin_swap_fees_summary.items():
    print(Fore.BLUE + f"{metric}: {value:.6f}")
print()

print(Fore.GREEN + "Sigmund Swap Fees Analysis Summary:")
for metric, value in sigmund_swap_fees_summary.items():
    print(Fore.BLUE + f"{metric}: {value:.6f}")
print()

print(Fore.GREEN + "Price Delta Analysis Summary:")
for metric, value in price_delta_sumary.items():
    print(Fore.BLUE + f"{metric}: {value:.6f}")
print()


print(Fore.GREEN + "Analysis Time: ", time.time() - start_time)

###################################################### CHARTS ######################################################

if not SHOW_PLOTS:
    exit()

# Calculate the absolute values of 'price_delta_percentage'
abs_price_delta_percentage = processed_df["price_delta_percentage"].abs()

# Determine the range for the x-axis
x_min = abs_price_delta_percentage.min()
x_max = abs_price_delta_percentage.max()

# Create bins: 5 bins per unit on the x-axis
bin_width = 0.2  # 1/5 unit for each bin
bins = np.arange(x_min, x_max + bin_width, bin_width)

# Create a figure with two subplots stacked horizontally
fig, axes = plt.subplots(1, 2, figsize=(18, 6))  # 1 row, 2 columns

# First Plot: Histogram of absolute 'price_delta_percentage' with 5 bins per unit
sns.histplot(abs_price_delta_percentage, bins=bins, kde=True, color="blue", stat="probability", ax=axes[0])
axes[0].set_xlabel("Absolute Price Delta Percentage")
axes[0].set_ylabel("Probability")
axes[0].set_title("Probability Distribution of Absolute Price Delta Percentage")

# Second Plot: Cumulative Probability Distribution of Absolute Price Delta Percentage
sns.histplot(
    abs_price_delta_percentage, bins=bins, kde=False, cumulative=True, color="red", stat="probability", ax=axes[1]
)
axes[1].set_xlabel("Absolute Price Delta Percentage")
axes[1].set_ylabel("Cumulative Probability")
axes[1].set_title("Cumulative Probability Distribution of Absolute Price Delta Percentage")

# Adjust layout to prevent overlapping
plt.tight_layout()

# Show the plots
plt.show()
