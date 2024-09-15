"""
Data Extraction Methodology using cryo:

1. Install cryo from: https://github.com/paradigmxyz/cryo

2. Run the following command to extract the data used in this analysis:
cryo logs \
    --label uniswap_v3_swaps \
    --blocks 20_020_000: \
    --reorg-buffer 1000 \
    --subdirs datatype \
    --contract 0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640 \
    --event-signature "Swap(address indexed sender, address indexed recipient, int256 amount0, int256 amount1, uint160 sqrtPriceX96, uint128 liquidity, int24 tick)" \
    --topic0 0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67 \
    --rpc <RPC_URL>

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
MAX_DATAPOINTS = 500_000  # Use this var to proccess only the last n entries when testing

# Sigmoid S-Curve Parameters 1
C0 = 0
C1 = 1
C2 = 600
C3 = 0.01

# Sigmoid S-Curve Parameters 2
# C0 = 0
# C1 = 1
# C2 = 800
# C3 = 0.0075

# Sigmoid S-Curve Parameters 3
# C0 = 0
# C1 = 1
# C2 = 1000
# C3 = 0.005


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


def calculate_sigmoid_fee(row: pd.Series) -> float:
    amount0 = int(row["event__amount0_string"])
    amount1 = int(row["event__amount1_string"])
    if amount0 < 0:
        return abs(row["dynamic_sigmoid_fee_percentage"] * amount0 * 10**-USDC_DECIMALS)
    return abs(row["dynamic_sigmoid_fee_percentage"] * amount1 * 10**-WETH_DECIMALS)


def convert_sigmoid_fee_to_usd(row: pd.Series) -> float:
    amount0 = int(row["event__amount0_string"])
    swap_fee = row["swap_fee_sigmoid"]
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
    exponent = C2 * (price_delta - C3)
    fee = C0 + C1 * expit(exponent)
    return fee


def is_arb_cancelled_by_sigmoid(row):
    condition1 = row["arb_profit"] > 0
    condition2 = row["arb_profit_sigmoid"] <= 0

    amount0 = int(row["event__amount0_string"])
    amount1 = int(row["event__amount1_string"])
    pool_price = row["price ETH/USD"]
    if amount0 < 0:
        swap_amount_usd = abs(amount0 * 10**-USDC_DECIMALS)  # assuming 1 usdc = 1 usd
    swap_amount_usd = abs(amount1 * 10**-WETH_DECIMALS) * pool_price
    # Assumption: A trader with a swap size >= 100,000 USD, trading in the arb direction is most likely an
    # informed trader and will not execute the trade in case of not having profit due to hogher fees
    condition3 = swap_amount_usd >= 100_000

    # Final decision based on all conditions
    return condition1 and condition2 and condition3


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

    # ########## DYNAMIC SIGMOID S-CURVE FEE ##########

    # # Set the first row value to 0.05% (or 0.0005 in decimal form)
    processed_df.loc[0, "dynamic_sigmoid_fee_percentage"] = SWAP_FEE

    last_computed_fee = SWAP_FEE

    for i in range(1, len(processed_df)):
        # Check if it's the first row with a given block number (TOP OF THE BLOCK)
        if processed_df.loc[i, "block_number"] != processed_df.loc[i - 1, "block_number"]:
            # Convert to float to ensure proper comparison
            price_delta = float(processed_df.loc[i, "price_delta_percentage"])
            abs_sig_fee_percentage = calculate_abs_sig_fee_percentage(abs(price_delta))
            if abs_sig_fee_percentage < 0:
                print("NEGATIVE FEE!!!!!!!", abs_sig_fee_percentage)
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
            processed_df.loc[i, "dynamic_sigmoid_fee_percentage"] = last_computed_fee
        else:
            # For all subsequent rows in the same block, use the already computed fee
            processed_df.loc[i, "dynamic_sigmoid_fee_percentage"] = last_computed_fee

    processed_df["swap_fee_sigmoid"] = processed_df.apply(calculate_sigmoid_fee, axis=1)
    processed_df["swap_fee_sigmoid_usd"] = processed_df.apply(convert_sigmoid_fee_to_usd, axis=1)

    # ###################################################### IL CALCULATION ######################################################

    # Get the first row value of 'price ETH/USD'
    first_price = processed_df.loc[0, "price ETH/USD"]

    # Calculate the ratio 'r' for each row and then calculate 'original_IL' using the formula IL = 1 - (2 * sqrt(r)) / (1 + r), where r = first_price / current_price
    # This formula assumes that liquidity was provided in the full range and was kept in the pool for the entire analysed period
    processed_df["original_IL_perc"] = (
        1
        - (2 * np.sqrt(first_price / processed_df["price ETH/USD"]))
        / (1 + (first_price / processed_df["price ETH/USD"]))
    ) * 100

    # ###################################################### FLAT FEE ARB PROFIT ######################################################

    # Convert 'event__amount1_string' to float (amount of ETH)
    processed_df["event__amount1_string"] = pd.to_numeric(processed_df["event__amount1_string"], errors="coerce")

    # Set swap fees and gas fees
    uniswap_fee_rate = SWAP_FEE  # 0.05%
    cex_fee_rate = 0.0001  # 0.01%
    gas_price_gwei = 3  # Avergae gas price in gwei
    gas_limit = 150000  # Gas limit for a swap transaction

    # Function to calculate arbitrage profit with fees
    def calculate_arb_profit_with_fees(row):
        uniswap_price = row["price ETH/USD"]
        cex_eth_price = float(row["price_usd"].replace(",", ""))
        amount_eth = abs(row["event__amount1_string"]) / 10**18  # Absolute value of ETH amount

        # Calculate the geometric mean pool price (modeling price impact of the trade)
        avg_pool_price = np.sqrt(uniswap_price * cex_eth_price)
        # avg_pool_price = uniswap_price

        # Calculate fees
        uniswap_fee = uniswap_fee_rate * avg_pool_price * amount_eth  # Uniswap fee in USD
        cex_fee = cex_fee_rate * cex_eth_price * amount_eth  # CEX fee in USD

        # Calculate gas fees in ETH and convert to USD
        gas_fee_eth = gas_price_gwei * gas_limit * 1e-9  # Convert gas fee to ETH (1 gwei = 10^-9 ETH)
        gas_fee_usd = gas_fee_eth * avg_pool_price  # Convert gas fee to USD using Uniswap price

        # Calculate arbitrage profit after deducting fees
        if (
            row["dynamic_sigmoid_fee_percentage"] > SWAP_FEE
        ):  # Swap occured in the direction of the arbitrage opportunity
            if uniswap_price < cex_eth_price:  # Buy on Uniswap, Sell on CEX
                arb_profit = (cex_eth_price - avg_pool_price) * amount_eth - uniswap_fee - cex_fee - gas_fee_usd
            elif uniswap_price > cex_eth_price:  # Sell on Uniswap, Buy on CEX
                arb_profit = (uniswap_price - avg_pool_price) * amount_eth - uniswap_fee - cex_fee - gas_fee_usd
            else:
                arb_profit = 0
        else:
            arb_profit = 0  # no arbitrage opportunity (price delta is zero)

        return arb_profit

    # Apply the function to each row to calculate 'arb_profit'
    processed_df["arb_profit"] = processed_df.apply(calculate_arb_profit_with_fees, axis=1)

    # ###################################################### SIGMOID FEE ARB PROFIT ######################################################

    # Set swap fees and gas fees
    cex_fee_rate = 0.0001  # 0.01%
    gas_price_gwei = 3  # Avergae gas price in gwei
    gas_limit = 150000  # Gas limit for a swap transaction

    # Function to calculate arbitrage profit with fees
    def calculate_arb_profit_with_sigmoid_fees(row):
        uniswap_price = row["price ETH/USD"]
        cex_eth_price = float(row["price_usd"].replace(",", ""))
        amount_eth = abs(row["event__amount1_string"]) / 10**18  # Absolute value of ETH amount

        sigmoid_fee_rate = row["dynamic_sigmoid_fee_percentage"]

        # Calculate the geometric mean pool price (modeling price impact of the trade)
        avg_pool_price = np.sqrt(uniswap_price * cex_eth_price)
        # avg_pool_price = uniswap_price

        # Calculate fees
        uniswap_fee = sigmoid_fee_rate * avg_pool_price * amount_eth  # Uniswap fee in USD
        cex_fee = cex_fee_rate * cex_eth_price * amount_eth  # CEX fee in USD

        # Calculate gas fees in ETH and convert to USD
        gas_fee_eth = gas_price_gwei * gas_limit * 1e-9  # Convert gas fee to ETH (1 gwei = 10^-9 ETH)
        gas_fee_usd = gas_fee_eth * avg_pool_price  # Convert gas fee to USD using Uniswap price

        # Calculate arbitrage profit after deducting fees
        if (
            row["dynamic_sigmoid_fee_percentage"] > SWAP_FEE
        ):  # Swap occured in the direction of the arbitrage opportunity
            if uniswap_price < cex_eth_price:  # Buy on Uniswap, Sell on CEX
                arb_profit = (cex_eth_price - avg_pool_price) * amount_eth - uniswap_fee - cex_fee - gas_fee_usd
            elif uniswap_price > cex_eth_price:  # Sell on Uniswap, Buy on CEX
                arb_profit = (uniswap_price - avg_pool_price) * amount_eth - uniswap_fee - cex_fee - gas_fee_usd
            else:
                arb_profit = 0
        else:
            arb_profit = 0  # no arbitrage opportunity

        return arb_profit

    # Apply the function to each row to calculate 'arb_profit'
    processed_df["arb_profit_sigmoid"] = processed_df.apply(calculate_arb_profit_with_sigmoid_fees, axis=1)

    processed_df["arb_cancelled_by_sigmoid"] = processed_df.apply(is_arb_cancelled_by_sigmoid, axis=1)

    processed_df.to_csv(OUTPUT_FILE_PATH, index=False)
else:

    print(Fore.GREEN + "No new rows to process.")


# ###################################################### TABLES ######################################################


pd.set_option("display.max_rows", 20)
print()
print(processed_df.head(100))
print()

# Flat Fee
total_swap_fee_usd = processed_df["swap_fee_usd"].sum()
mean_il_perc = processed_df["original_IL_perc"].mean()
median_il_perc = processed_df["original_IL_perc"].median()
mean_swap_fee_usd = processed_df["swap_fee_usd"].mean()
median_swap_fee_usd = processed_df["swap_fee_usd"].median()
std_swap_fee_usd = processed_df["swap_fee_usd"].std()
min_swap_fee_usd = processed_df["swap_fee_usd"].min()
max_swap_fee_usd = processed_df["swap_fee_usd"].max()
total_arb_profit = processed_df.loc[processed_df["arb_profit"] > 0, "arb_profit"].sum()
total_arb_profit_sig = processed_df.loc[processed_df["arb_profit_sigmoid"] > 0, "arb_profit_sigmoid"].sum()


flat_swap_fees_summary = {
    "Total Swap Fee USD": total_swap_fee_usd,
    "Mean IL Percentage": mean_il_perc,
    "Median IL Percentage": median_il_perc,
    "Median Swap Fee USD": median_swap_fee_usd,
    "Mean Swap Fee USD": mean_swap_fee_usd,
    "Std Dev Swap Fee USD": std_swap_fee_usd,
    "Min Swap Fee USD": min_swap_fee_usd,
    "Max Swap Fee USD": max_swap_fee_usd,
    "Total Arb Profit": total_arb_profit,
}


# sigmoid Fee
total_swap_fee_sigmoid_usd = processed_df["swap_fee_sigmoid_usd"].sum()
mean_swap_fee_sigmoid_usd = processed_df["swap_fee_sigmoid_usd"].mean()
median_swap_fee_sigmoid_usd = processed_df["swap_fee_sigmoid_usd"].median()
std_swap_fee_sigmoid_usd = processed_df["swap_fee_sigmoid_usd"].std()
min_swap_fee_sigmoid_usd = processed_df["swap_fee_sigmoid_usd"].min()
max_swap_fee_sigmoid_usd = processed_df["swap_fee_sigmoid_usd"].max()
total_avoided_arb_trades = int(processed_df["arb_cancelled_by_sigmoid"].sum())

sigmoid_swap_fees_summary = {
    "Total Swap Fee USD": total_swap_fee_sigmoid_usd,
    "Median Swap Fee USD": median_swap_fee_sigmoid_usd,
    "Mean Swap Fee USD": mean_swap_fee_sigmoid_usd,
    "Std Dev Swap Fee USD": std_swap_fee_sigmoid_usd,
    "Min Swap Fee USD": min_swap_fee_sigmoid_usd,
    "Max Swap Fee USD": max_swap_fee_sigmoid_usd,
    "Total Arb Profit": total_arb_profit_sig,
    "Total Avoided Arb Trades": total_avoided_arb_trades,
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


print(Fore.GREEN + "sigmoid Swap Fees Analysis Summary:")
for metric, value in sigmoid_swap_fees_summary.items():
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
# x_max = abs_price_delta_percentage.max()
x_max = 10

# Create bins: 5 bins per unit on the x-axis
bin_width = 0.2  # 1/5 unit for each bin
bins = np.arange(x_min, x_max + bin_width, bin_width)

# Create a figure with two subplots stacked horizontally
fig, axes = plt.subplots(1, 2, figsize=(18, 6))  # 1 row, 2 columns

# First Plot: Histogram of absolute 'price_delta_percentage' with 5 bins per unit
sns.histplot(abs_price_delta_percentage, bins=bins, kde=False, color="blue", stat="probability", ax=axes[0])
axes[0].set_xlabel("Absolute Price Delta Percentage")
axes[0].set_ylabel("Probability")
axes[0].set_title("Pool/Binance Price Delta Probability Distribution")

axes[0].set_xticks(np.arange(x_min, x_max + 1, 1))

# Second Plot: Cumulative Probability Distribution of Absolute Price Delta Percentage
sns.histplot(
    abs_price_delta_percentage, bins=bins, kde=False, cumulative=True, color="red", stat="probability", ax=axes[1]
)
axes[1].set_xlabel("Absolute Price Delta Percentage")
axes[1].set_ylabel("Cumulative Probability")
axes[1].set_title("Pool/Binance Cumulative Probability Distribution")

axes[1].set_xticks(np.arange(x_min, x_max + 1, 1))

# Adjust layout to prevent overlapping
plt.tight_layout()

# Show the plots
plt.show()
