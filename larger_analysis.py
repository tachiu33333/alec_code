import requests
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import timedelta
import time

COMPANIES = {
    "LMT": "Lockheed Martin",
    "RTX": "Raytheon",
    "BA":  "Boeing",
    "NOC": "Northrop Grumman",
    "GD":  "General Dynamics",
    "BAH": "Booz Allen Hamilton",
}

EVENT_WINDOW_DAYS    = 5
MIN_MOD_AMOUNT       = 10_000_000
MAX_AWARDS_PER_CO    = 20
START_DATE           = "2022-01-01"
END_DATE             = "2023-12-31"

def fetch_award_ids(company_name: str, ticker: str) -> list:
    print(f"  [{ticker}] Searching awards for '{company_name}'...")

    url = "https://api.usaspending.gov/api/v2/search/spending_by_award/"
    payload = {
        "filters": {
            "recipient_search_text": [company_name],
            "award_type_codes": ["A", "B", "C", "D"],
            "time_period": [{"start_date": START_DATE, "end_date": END_DATE}],
        },
        "fields": ["Award ID", "generated_internal_id", "Award Amount",
                   "Recipient Name", "Start Date", "Awarding Agency"],
        "sort": "Award Amount",
        "order": "desc",
        "limit": MAX_AWARDS_PER_CO,
        "page": 1,
        "subawards": False,
    }

    try:
        r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
        results = r.json().get("results", [])
        award_ids = [a["generated_internal_id"] for a in results if a.get("generated_internal_id")]
        print(f"    Found {len(award_ids)} awards")
        return award_ids
    except Exception as e:
        print(f"    Error fetching awards: {e}")
        return []

def fetch_transactions_for_award(award_id: str) -> pd.DataFrame:
    url = "https://api.usaspending.gov/api/v2/transactions/"
    payload = {
        "award_id": award_id,
        "page": 1,
        "sort": "action_date",
        "order": "asc",
        "limit": 5000,
    }

    try:
        r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            return pd.DataFrame()
        df = pd.DataFrame(results)
        df["award_id"] = award_id
        return df
    except Exception:
        return pd.DataFrame()

def fetch_all_modifications(award_ids: list, ticker: str) -> pd.DataFrame:
    all_transactions = []

    for award_id in award_ids:
        df = fetch_transactions_for_award(award_id)
        if df.empty:
            continue

        if "modification_number" in df.columns:
            df = df[df["modification_number"] != "0"]

        if "federal_action_obligation" in df.columns:
            df["federal_action_obligation"] = pd.to_numeric(
                df["federal_action_obligation"], errors="coerce"
            )
            df = df[df["federal_action_obligation"].abs() >= MIN_MOD_AMOUNT]

        if not df.empty:
            all_transactions.append(df)

        time.sleep(0.3)

    if not all_transactions:
        return pd.DataFrame()

    combined = pd.concat(all_transactions, ignore_index=True)
    combined["ticker"] = ticker

    if "action_date" in combined.columns:
        combined["action_date"] = pd.to_datetime(combined["action_date"])

    print(f"    {len(combined)} significant modifications (>= ${MIN_MOD_AMOUNT:,.0f})")
    return combined

def fetch_stock_prices(ticker: str) -> pd.DataFrame:
    print(f"  [{ticker}] Fetching stock prices...")
    start = (pd.to_datetime(START_DATE) - timedelta(days=15)).strftime("%Y-%m-%d")
    end   = (pd.to_datetime(END_DATE)   + timedelta(days=15)).strftime("%Y-%m-%d")

    hist = yf.Ticker(ticker).history(start=start, end=end)
    df = hist[["Close"]].reset_index()
    df.columns = ["date", "close"]
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df.sort_values("date").reset_index(drop=True)

def fetch_sp500() -> pd.DataFrame:
    print("  Fetching S&P 500 benchmark...")
    start = (pd.to_datetime(START_DATE) - timedelta(days=15)).strftime("%Y-%m-%d")
    end   = (pd.to_datetime(END_DATE)   + timedelta(days=15)).strftime("%Y-%m-%d")

    hist = yf.Ticker("^GSPC").history(start=start, end=end)
    df = hist[["Close"]].reset_index()
    df.columns = ["date", "close"]
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df.sort_values("date").reset_index(drop=True)

def get_price_after_n_days(event_date, prices: pd.DataFrame, n: int):
    future = prices[prices["date"] >= event_date]
    if len(future) < n + 1:
        return None, None, None
    return future.iloc[0]["date"], future.iloc[0]["close"], future.iloc[min(n, len(future)-1)]["close"]

def compute_event_returns(modifications: pd.DataFrame,
                           stock_prices: pd.DataFrame,
                           sp500_prices: pd.DataFrame) -> pd.DataFrame:
    results = []

    for _, row in modifications.iterrows():
        event_date = row.get("action_date")
        if pd.isnull(event_date):
            continue

        t0_date, p0, p1 = get_price_after_n_days(event_date, stock_prices, EVENT_WINDOW_DAYS)
        if p0 is None:
            continue

        _, sp0, sp1 = get_price_after_n_days(event_date, sp500_prices, EVENT_WINDOW_DAYS)
        if sp0 is None:
            continue

        mod_amt    = row.get("federal_action_obligation", 0)
        stock_ret  = (p1 - p0) / p0
        market_ret = (sp1 - sp0) / sp0
        excess_ret = stock_ret - market_ret

        results.append({
            "ticker":        row["ticker"],
            "event_date":    event_date,
            "trade_date":    t0_date,
            "mod_amount":    mod_amt,
            "award_id":      row.get("award_id", ""),
            "description":   row.get("description", ""),
            "mod_number":    row.get("modification_number", ""),
            "price_t0":      p0,
            "price_t1":      p1,
            "stock_return":  stock_ret,
            "market_return": market_ret,
            "excess_return": excess_ret,
            "direction":     "increase" if mod_amt > 0 else "decrease",
        })

    return pd.DataFrame(results)

def run_hypothesis_test(results: pd.DataFrame):
    print("\n" + "═" * 55)
    print("HYPOTHESIS TEST RESULTS")
    print(f"Event window: {EVENT_WINDOW_DAYS} trading days")
    print(f"Min modification size: ${MIN_MOD_AMOUNT:,.0f}")
    print("═" * 55)

    increases = results[results["direction"] == "increase"]["excess_return"].dropna()
    decreases = results[results["direction"] == "decrease"]["excess_return"].dropna()

    for label, group in [("CONTRACT INCREASES", increases), ("CONTRACT DECREASES", decreases)]:
        print(f"\n── {label}  (n={len(group)}) ──")
        if len(group) == 0:
            print("  No data")
            continue
        print(f"  Mean excess return:   {group.mean()*100:+.2f}%")
        print(f"  Median excess return: {group.median()*100:+.2f}%")
        print(f"  Std deviation:        {group.std()*100:.2f}%")
        if len(group) > 1:
            t, p = stats.ttest_1samp(group, 0)
            sig = "✓ SIGNIFICANT" if p < 0.05 else "✗ not significant"
            print(f"  T-test vs 0:   t={t:.3f}, p={p:.4f}  {sig}")

    if len(increases) > 1 and len(decreases) > 1:
        t, p = stats.ttest_ind(increases, decreases)
        sig = "✓ SIGNIFICANT" if p < 0.05 else "✗ not significant"
        print(f"\n── INCREASES vs DECREASES ──")
        print(f"  T-test:  t={t:.3f}, p={p:.4f}  {sig}")

def plot_results(results: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Gov Contract Modifications → {EVENT_WINDOW_DAYS}-Day Excess Stock Returns\n"
        f"(Market-adjusted, min modification = ${MIN_MOD_AMOUNT/1e6:.0f}M)",
        fontsize=13
    )

    palette = {"increase": "

    ax = axes[0, 0]
    sns.boxplot(data=results, x="direction", y="excess_return", palette=palette, ax=ax)
    ax.axhline(0, color="black", linestyle="--", lw=1)
    ax.set_title("Excess Return by Contract Direction")
    ax.set_ylabel("Excess Return (stock − S&P 500)")
    ax.set_xlabel("")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.1f}%"))

    ax = axes[0, 1]
    colors = results["direction"].map(palette)
    ax.scatter(results["mod_amount"] / 1e6, results["excess_return"],
               c=colors, alpha=0.6, edgecolors="white", linewidth=0.5, s=60)
    ax.axhline(0, color="black", linestyle="--", lw=1)
    ax.axvline(0, color="black", linestyle="--", lw=1)
    ax.set_title("Modification Size vs. Excess Return")
    ax.set_xlabel("Modification Amount ($M)")
    ax.set_ylabel("Excess Return")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.1f}%"))

    ax = axes[1, 0]
    ticker_means = (
        results.groupby(["ticker", "direction"])["excess_return"]
        .mean().unstack(fill_value=0)
    )
    ticker_means.plot(kind="bar", ax=ax, color=["
    ax.axhline(0, color="black", linestyle="--", lw=1)
    ax.set_title("Mean Excess Return by Company")
    ax.set_ylabel("Mean Excess Return")
    ax.set_xlabel("")
    ax.legend(title="Direction")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.1f}%"))
    plt.setp(ax.get_xticklabels(), rotation=30)

    ax = axes[1, 1]
    for direction, color in palette.items():
        subset = results[results["direction"] == direction]["excess_return"]
        if not subset.empty:
            sns.kdeplot(subset * 100, ax=ax, label=direction, color=color, fill=True, alpha=0.3)
    ax.axvline(0, color="black", linestyle="--", lw=1)
    ax.set_title("Distribution of Excess Returns")
    ax.set_xlabel("Excess Return (%)")
    ax.legend()

    plt.tight_layout()
    plt.savefig("contract_stock_results.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved → contract_stock_results.png")
    plt.show()

def main():
    sp500 = fetch_sp500()
    all_results = []

    for ticker, company_name in COMPANIES.items():
        print(f"\n{'─'*55}")
        print(f"Processing {ticker} — {company_name}")

        award_ids = fetch_award_ids(company_name, ticker)
        if not award_ids:
            continue

        modifications = fetch_all_modifications(award_ids, ticker)
        if modifications.empty:
            print(f"  No significant modifications found for {ticker}")
            continue

        try:
            stock_prices = fetch_stock_prices(ticker)
        except Exception as e:
            print(f"  Could not fetch stock prices for {ticker}: {e}")
            continue

        results = compute_event_returns(modifications, stock_prices, sp500)
        if not results.empty:
            all_results.append(results)
            print(f"  Matched {len(results)} events to stock returns")

    if not all_results:
        print("\nNo results found. Check your internet connection and API access.")
        return

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv("contract_stock_results.csv", index=False)
    print(f"\n✓ Saved {len(combined)} total events → contract_stock_results.csv")

    print("\n── Events per company ──")
    print(combined.groupby(["ticker", "direction"]).size().unstack(fill_value=0).to_string())

    run_hypothesis_test(combined)
    plot_results(combined)

if __name__ == "__main__":
    main()
