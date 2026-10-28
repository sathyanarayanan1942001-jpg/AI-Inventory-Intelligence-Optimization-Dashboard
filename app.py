"""
AI Inventory Intelligence & Optimization Dashboard
A transparent, rule-based inventory analytics and decision support system.
Suitable for academic demonstration and retail inventory optimization.
"""

import math
import numpy as np
import pandas as pd
import streamlit as st

# ==============================================================================
# 1. PAGE CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(
    page_title="AI Inventory Intelligence & Optimization",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# 2. DETERMINISTIC SAMPLE DATA GENERATION
# ==============================================================================
@st.cache_data
def generate_inventory_data():
    """
    Generates deterministic synthetic data for 30 retail products across 5 categories
    with 12 months of sales history and realistic supply chain attributes.
    """
    np.random.seed(42)

    products = [
        # Groceries (Fast moving, perishable/staple, shorter lead times)
        {"id": "PRD-001", "name": "Organic Whole Milk (1 Gallon)", "category": "Groceries", "base_sales": 280, "season": "Stable / None", "lead_time": 4, "supplier": "Valley Fresh Farms", "cost": 2.40, "price": 4.50, "moq": 50, "reliability": 0.96, "stock_bias": "critically_low"},
        {"id": "PRD-002", "name": "Basmati Rice (5 kg)", "category": "Groceries", "base_sales": 150, "season": "Stable / None", "lead_time": 10, "supplier": "AgroGlobal Mills", "cost": 8.50, "price": 14.99, "moq": 40, "reliability": 0.94, "stock_bias": "healthy"},
        {"id": "PRD-003", "name": "Extra Virgin Olive Oil (1L)", "category": "Groceries", "base_sales": 90, "season": "Holiday Peak", "lead_time": 14, "supplier": "Mediterranean Harvest", "cost": 9.20, "price": 18.50, "moq": 30, "reliability": 0.91, "stock_bias": "healthy"},
        {"id": "PRD-004", "name": "Artisan Ground Coffee (500g)", "category": "Groceries", "base_sales": 220, "season": "Winter Peak", "lead_time": 7, "supplier": "Summit Roasters", "cost": 6.80, "price": 13.99, "moq": 60, "reliability": 0.95, "stock_bias": "critically_low"},
        {"id": "PRD-005", "name": "Artisanal Sourdough Bread", "category": "Groceries", "base_sales": 310, "season": "Stable / None", "lead_time": 2, "supplier": "Golden Crust Bakery", "cost": 1.60, "price": 3.99, "moq": 40, "reliability": 0.98, "stock_bias": "reorder"},
        {"id": "PRD-006", "name": "Premium Canned Truffle Sauce", "category": "Groceries", "base_sales": 15, "season": "Holiday Peak", "lead_time": 25, "supplier": "EuroGourmet Imports", "cost": 18.00, "price": 34.00, "moq": 20, "reliability": 0.88, "stock_bias": "excess"},

        # Electronics (High value, longer lead times, tech volatility)
        {"id": "PRD-007", "name": "Wireless Noise-Canceling Earbuds", "category": "Electronics", "base_sales": 160, "season": "Holiday Peak", "lead_time": 30, "supplier": "NexTech Pacific", "cost": 32.00, "price": 69.99, "moq": 50, "reliability": 0.87, "stock_bias": "critically_low"},
        {"id": "PRD-008", "name": "USB-C Fast Charger (65W GaN)", "category": "Electronics", "base_sales": 190, "season": "Stable / None", "lead_time": 21, "supplier": "VoltPower Tech", "cost": 11.50, "price": 25.99, "moq": 80, "reliability": 0.93, "stock_bias": "reorder"},
        {"id": "PRD-009", "name": "Braided 4K HDMI Cable (2m)", "category": "Electronics", "base_sales": 75, "season": "Stable / None", "lead_time": 28, "supplier": "NexTech Pacific", "cost": 3.80, "price": 12.50, "moq": 100, "reliability": 0.90, "stock_bias": "excess"},
        {"id": "PRD-010", "name": "RGB Mechanical Gaming Keyboard", "category": "Electronics", "base_sales": 65, "season": "Holiday Peak", "lead_time": 35, "supplier": "K-Tronix Shenzen", "cost": 29.00, "price": 64.99, "moq": 30, "reliability": 0.82, "stock_bias": "reorder"},
        {"id": "PRD-011", "name": "Smart Water-Resistant Fitness Band", "category": "Electronics", "base_sales": 110, "season": "Winter Peak", "lead_time": 25, "supplier": "NexTech Pacific", "cost": 22.00, "price": 49.99, "moq": 45, "reliability": 0.89, "stock_bias": "healthy"},
        {"id": "PRD-012", "name": "Legacy Micro-USB Dock Station", "category": "Electronics", "base_sales": 12, "season": "Stable / None", "lead_time": 40, "supplier": "K-Tronix Shenzen", "cost": 14.00, "price": 29.99, "moq": 50, "reliability": 0.84, "stock_bias": "excess"},

        # Personal Care (Moderate demand, steady replenishment)
        {"id": "PRD-013", "name": "Broad Spectrum Mineral Sunscreen SPF50", "category": "Personal Care", "base_sales": 180, "season": "Summer Peak", "lead_time": 14, "supplier": "DermaPure Labs", "cost": 6.50, "price": 15.99, "moq": 50, "reliability": 0.94, "stock_bias": "critically_low"},
        {"id": "PRD-014", "name": "Botanical Sulfate-Free Shampoo (400ml)", "category": "Personal Care", "base_sales": 140, "season": "Stable / None", "lead_time": 10, "supplier": "Naturals BioCorp", "cost": 4.20, "price": 10.99, "moq": 60, "reliability": 0.96, "stock_bias": "healthy"},
        {"id": "PRD-015", "name": "Activated Charcoal Whitening Toothpaste", "category": "Personal Care", "base_sales": 165, "season": "Stable / None", "lead_time": 8, "supplier": "Naturals BioCorp", "cost": 2.10, "price": 5.99, "moq": 100, "reliability": 0.97, "stock_bias": "reorder"},
        {"id": "PRD-016", "name": "Hydrating Hyaluronic Facial Cleanser", "category": "Personal Care", "base_sales": 125, "season": "Winter Peak", "lead_time": 12, "supplier": "DermaPure Labs", "cost": 5.10, "price": 13.50, "moq": 40, "reliability": 0.93, "stock_bias": "healthy"},
        {"id": "PRD-017", "name": "Sonic Electric Toothbrush Replacement Heads", "category": "Personal Care", "base_sales": 95, "season": "Stable / None", "lead_time": 18, "supplier": "VoltPower Tech", "cost": 5.80, "price": 16.99, "moq": 50, "reliability": 0.92, "stock_bias": "healthy"},
        {"id": "PRD-018", "name": "Glitter Body Shimmer Spray", "category": "Personal Care", "base_sales": 18, "season": "Holiday Peak", "lead_time": 20, "supplier": "DermaPure Labs", "cost": 4.00, "price": 12.00, "moq": 60, "reliability": 0.89, "stock_bias": "excess"},

        # Clothing (Seasonal swings, sizing, long lead times)
        {"id": "PRD-019", "name": "Men's Classic Cotton Crew T-Shirt (3-Pack)", "category": "Clothing", "base_sales": 210, "season": "Summer Peak", "lead_time": 22, "supplier": "Apex Garments Ltd", "cost": 9.00, "price": 22.50, "moq": 75, "reliability": 0.90, "stock_bias": "critically_low"},
        {"id": "PRD-020", "name": "Thermal Heavyweight Fleece Jacket", "category": "Clothing", "base_sales": 130, "season": "Winter Peak", "lead_time": 35, "supplier": "Nordic Apparel Co", "cost": 28.00, "price": 68.00, "moq": 35, "reliability": 0.86, "stock_bias": "reorder"},
        {"id": "PRD-021", "name": "Relaxed Fit Stretch Denim Jeans", "category": "Clothing", "base_sales": 115, "season": "Stable / None", "lead_time": 28, "supplier": "Apex Garments Ltd", "cost": 17.50, "price": 42.00, "moq": 40, "reliability": 0.91, "stock_bias": "healthy"},
        {"id": "PRD-022", "name": "Merino Wool Anti-Blister Hiking Socks", "category": "Clothing", "base_sales": 85, "season": "Winter Peak", "lead_time": 18, "supplier": "Nordic Apparel Co", "cost": 5.20, "price": 14.50, "moq": 50, "reliability": 0.94, "stock_bias": "healthy"},
        {"id": "PRD-023", "name": "Lightweight Breathable Running Shorts", "category": "Clothing", "base_sales": 145, "season": "Summer Peak", "lead_time": 24, "supplier": "Apex Garments Ltd", "cost": 8.00, "price": 21.00, "moq": 60, "reliability": 0.88, "stock_bias": "reorder"},
        {"id": "PRD-024", "name": "Novelty Holiday Pattern Ugly Sweater", "category": "Clothing", "base_sales": 22, "season": "Holiday Peak", "lead_time": 30, "supplier": "Nordic Apparel Co", "cost": 16.00, "price": 38.00, "moq": 40, "reliability": 0.85, "stock_bias": "excess"},

        # Home Essentials (Utility goods, bulk purchasing)
        {"id": "PRD-025", "name": "Microfiber Multi-Surface Cleaning Cloths (12pk)", "category": "Home Essentials", "base_sales": 230, "season": "Stable / None", "lead_time": 9, "supplier": "CleanHome Corp", "cost": 3.60, "price": 9.99, "moq": 100, "reliability": 0.97, "stock_bias": "critically_low"},
        {"id": "PRD-026", "name": "Ultra Concentrated Dish Soap (1L)", "category": "Home Essentials", "base_sales": 260, "season": "Stable / None", "lead_time": 6, "supplier": "CleanHome Corp", "cost": 1.90, "price": 4.79, "moq": 120, "reliability": 0.98, "stock_bias": "reorder"},
        {"id": "PRD-027", "name": "Insulated Stainless Steel Water Bottle (1L)", "category": "Home Essentials", "base_sales": 100, "season": "Summer Peak", "lead_time": 25, "supplier": "VoltPower Tech", "cost": 8.50, "price": 22.00, "moq": 50, "reliability": 0.91, "stock_bias": "healthy"},
        {"id": "PRD-028", "name": "Organic Aromatherapy Soy Candle", "category": "Home Essentials", "base_sales": 70, "season": "Holiday Peak", "lead_time": 15, "supplier": "Summit Roasters", "cost": 6.20, "price": 16.50, "moq": 40, "reliability": 0.93, "stock_bias": "healthy"},
        {"id": "PRD-029", "name": "Large Antimicrobial Bamboo Cutting Board", "category": "Home Essentials", "base_sales": 55, "season": "Stable / None", "lead_time": 20, "supplier": "AgroGlobal Mills", "cost": 7.40, "price": 19.99, "moq": 35, "reliability": 0.92, "stock_bias": "healthy"},
        {"id": "PRD-030", "name": "Ceramic Decorative Potpourri Bowl", "category": "Home Essentials", "base_sales": 10, "season": "Stable / None", "lead_time": 30, "supplier": "CleanHome Corp", "cost": 12.00, "price": 27.00, "moq": 30, "reliability": 0.89, "stock_bias": "excess"},
    ]

    months = [f"M{str(i).zfill(2)}" for i in range(1, 13)]
    records = []

    for item in products:
        base = item["base_sales"]
        season = item["season"]
        sales_history = []

        for m_idx in range(1, 13):
            noise = np.random.normal(1.0, 0.08)
            seasonal_mult = 1.0

            # Seasonality models across the 12 months
            if season == "Summer Peak":
                # Peak in months 5, 6, 7 (e.g. May, Jun, Jul)
                if m_idx in [5, 6, 7]:
                    seasonal_mult = 1.55
                elif m_idx in [4, 8]:
                    seasonal_mult = 1.25
                else:
                    seasonal_mult = 0.80
            elif season == "Winter Peak":
                # Peak in months 10, 11, 12, 1, 2
                if m_idx in [11, 12, 1]:
                    seasonal_mult = 1.60
                elif m_idx in [10, 2]:
                    seasonal_mult = 1.25
                else:
                    seasonal_mult = 0.75
            elif season == "Holiday Peak":
                # Peak in months 11, 12
                if m_idx in [11, 12]:
                    seasonal_mult = 1.75
                elif m_idx == 10:
                    seasonal_mult = 1.20
                else:
                    seasonal_mult = 0.85

            monthly_val = max(3, int(round(base * seasonal_mult * noise)))
            sales_history.append(monthly_val)

        # Baseline average daily sales for inventory setup
        recent_daily_rate = max(0.2, np.mean(sales_history[-3:]) / 30.0)
        lead_time = item["lead_time"]

        # Synthetic current inventory assignment based on archetype
        bias = item["stock_bias"]
        if bias == "critically_low":
            # Stock is less than lead-time demand (imminent stockout risk)
            cur_stock = max(1, int(round(recent_daily_rate * lead_time * 0.4)))
        elif bias == "reorder":
            # Stock is around or slightly under reorder point
            cur_stock = int(round(recent_daily_rate * lead_time * 1.1))
        elif bias == "healthy":
            # Stock is comfortable (~25-40 days of coverage)
            cur_stock = int(round(recent_daily_rate * (lead_time + 25)))
        else:  # excess
            # Stock is excessively high (~90-150 days of coverage)
            cur_stock = int(round(recent_daily_rate * (lead_time + 110) + item["moq"]))

        rec = {
            "product_id": item["id"],
            "product_name": item["name"],
            "category": item["category"],
            "current_inventory": cur_stock,
            "lead_time_days": item["lead_time"],
            "supplier_name": item["supplier"],
            "unit_cost": item["cost"],
            "unit_price": item["price"],
            "min_order_qty": item["moq"],
            "supplier_reliability": item["reliability"],
            "seasonal_indicator": item["season"],
            "monthly_sales": sales_history,
        }
        for i, m_name in enumerate(months):
            rec[m_name] = sales_history[i]

        records.append(rec)

    return pd.DataFrame(records)


# ==============================================================================
# 3. STATISTICAL DEMAND & INVENTORY ENGINE
# ==============================================================================
def calculate_inventory_metrics(df: pd.DataFrame, target_coverage_days: int = 30, review_period_days: int = 14) -> pd.DataFrame:
    """
    Computes statistical demand forecasts, safety stocks, reorder points,
    stockout risks, and excess stock values without black-box ML.
    """
    res = df.copy()
    month_cols = [f"M{str(i).zfill(2)}" for i in range(1, 13)]

    # 1. Historical 12-month metrics
    res["annual_sales"] = res[month_cols].sum(axis=1)
    res["avg_daily_demand"] = res["annual_sales"] / 365.0

    # 2. Weighted Moving Average (WMA) of recent 4 months (M09, M10, M11, M12)
    # Weights give higher emphasis to recent patterns: 10%, 20%, 30%, 40%
    weights = [0.10, 0.20, 0.30, 0.40]
    res["wma_recent_monthly"] = (
        res["M09"] * weights[0]
        + res["M10"] * weights[1]
        + res["M11"] * weights[2]
        + res["M12"] * weights[3]
    )
    # Seasonal adjustment factor: compare M12 with 12M average
    avg_m = res["annual_sales"] / 12.0
    res["seasonality_ratio"] = np.where(avg_m > 0, res["M12"] / avg_m, 1.0)
    
    # Moderate seasonal adjustment factor (bounded to avoid over-amplification)
    res["seasonal_multiplier"] = np.clip(res["seasonality_ratio"], 0.85, 1.30)
    
    # Near-term daily demand forecast
    res["forecast_daily_demand"] = (res["wma_recent_monthly"] / 30.0) * (0.8 + 0.2 * res["seasonal_multiplier"])
    res["forecast_daily_demand"] = res["forecast_daily_demand"].apply(lambda x: max(0.1, round(x, 2)))

    # 3. Demand Trend (% change of last 3 months vs previous 3 months)
    recent_q = res[["M10", "M11", "M12"]].mean(axis=1)
    prior_q = res[["M07", "M08", "M09"]].mean(axis=1)
    res["demand_trend_pct"] = np.where(
        prior_q > 0,
        ((recent_q - prior_q) / prior_q) * 100.0,
        0.0
    ).round(1)

    # 4. Demand Variability (Monthly Std Dev scaled to Daily)
    monthly_std = res[month_cols].std(axis=1)
    res["daily_demand_std"] = (monthly_std / np.sqrt(30.0)).apply(lambda x: max(0.05, round(x, 2)))

    # 5. Safety Stock calculation
    # Z = 1.65 for 95% cycle service level
    # Reliability penalty: if reliability is below 95%, scale safety buffer up
    z_score = 1.65
    res["reliability_penalty"] = res["supplier_reliability"].apply(lambda r: 1.0 + max(0.0, (0.95 - r) * 2.0))
    res["safety_stock"] = (
        z_score * res["daily_demand_std"] * np.sqrt(res["lead_time_days"]) * res["reliability_penalty"]
    ).apply(lambda x: int(math.ceil(x)))

    # 6. Lead Time Demand & Reorder Point (ROP)
    res["lead_time_demand"] = (res["forecast_daily_demand"] * res["lead_time_days"]).apply(lambda x: int(math.ceil(x)))
    res["reorder_point"] = res["lead_time_demand"] + res["safety_stock"]

    # 7. Days of Inventory Remaining (safe division)
    res["days_of_stock_remaining"] = np.where(
        res["forecast_daily_demand"] > 0,
        (res["current_inventory"] / res["forecast_daily_demand"]).round(1),
        999.0
    )

    # 8. Target Stock Level & Excess Stock
    # Target stock covers lead time + review period + safety stock buffer
    res["target_stock"] = (
        (res["forecast_daily_demand"] * (res["lead_time_days"] + review_period_days)) + res["safety_stock"]
    ).apply(lambda x: int(math.ceil(x)))

    # Max acceptable operational stock limit based on target coverage slider
    res["target_coverage_stock"] = (
        (res["forecast_daily_demand"] * target_coverage_days) + res["safety_stock"]
    ).apply(lambda x: int(math.ceil(x)))

    res["excess_units"] = (res["current_inventory"] - res["target_coverage_stock"]).apply(lambda x: max(0, int(x)))
    res["excess_inventory_value"] = (res["excess_units"] * res["unit_cost"]).round(2)
    res["total_inventory_value"] = (res["current_inventory"] * res["unit_cost"]).round(2)

    # 9. Inventory Health Classification
    def classify_status(row):
        # Stockout risk: current stock cannot even survive lead time demand
        if row["current_inventory"] < row["lead_time_demand"]:
            return "STOCKOUT RISK"
        # Reorder now: current stock has breached or met the reorder point
        elif row["current_inventory"] <= row["reorder_point"]:
            return "REORDER NOW"
        # Excess stock: current inventory significantly exceeds policy target
        elif row["current_inventory"] > row["target_coverage_stock"]:
            return "EXCESS STOCK"
        # Otherwise healthy operating buffer
        else:
            return "HEALTHY"

    res["inventory_status"] = res.apply(classify_status, axis=1)

    # 10. Suggested Order Quantity & Purchasing Rationale
    def calc_reorder(row):
        if row["current_inventory"] <= row["reorder_point"]:
            deficit = row["target_stock"] - row["current_inventory"]
            if deficit > 0:
                # Must respect Minimum Order Quantity (MOQ)
                suggested = max(int(math.ceil(deficit)), int(row["min_order_qty"]))
                cost = round(suggested * row["unit_cost"], 2)
                
                reasons = []
                if row["current_inventory"] < row["lead_time_demand"]:
                    reasons.append(f"Critical: Stock ({row['current_inventory']}) < Lead-time demand ({row['lead_time_demand']}).")
                else:
                    reasons.append(f"Stock ({row['current_inventory']}) ≤ ROP ({row['reorder_point']}).")

                if suggested == row["min_order_qty"]:
                    reasons.append(f"Adjusted up to Supplier MOQ ({row['min_order_qty']} units).")
                else:
                    reasons.append(f"Covers {review_period_days}d review + {row['lead_time_days']}d lead time.")
                
                return suggested, cost, " ".join(reasons)
        return 0, 0.0, "Inventory above reorder point; no replenishment needed."

    reorder_tuples = res.apply(calc_reorder, axis=1)
    res["suggested_order_qty"] = [t[0] for t in reorder_tuples]
    res["estimated_order_cost"] = [t[1] for t in reorder_tuples]
    res["reorder_reason"] = [t[2] for t in reorder_tuples]

    return res


# ==============================================================================
# 4. RULE-BASED PRESCRIPTIVE RECOMMENDATION ENGINE
# ==============================================================================
def generate_prescriptive_insights(df: pd.DataFrame) -> list:
    """
    Translates inventory metrics into transparent, auditable business recommendations.
    Every recommendation explains the identified issue, supporting math, and suggested action.
    """
    recommendations = []

    for _, row in df.iterrows():
        p_name = row["product_name"]
        cat = row["category"]
        cur = row["current_inventory"]
        daily_d = row["forecast_daily_demand"]
        lead_t = row["lead_time_days"]
        ltd = row["lead_time_demand"]
        rop = row["reorder_point"]
        days_rem = row["days_of_stock_remaining"]
        rel = row["supplier_reliability"]
        trend = row["demand_trend_pct"]
        excess = row["excess_units"]
        ex_val = row["excess_inventory_value"]
        order_qty = row["suggested_order_qty"]
        status = row["inventory_status"]

        # Rule 1: High Demand + Imminent Stockout
        if status == "STOCKOUT RISK":
            recommendations.append({
                "Product": p_name,
                "Category": cat,
                "Priority": "CRITICAL",
                "Identified Issue": f"Stockout Risk: Only {days_rem:.1f} days of stock left vs {lead_t} days supplier lead time.",
                "Calculations": f"Current Stock: {cur} units | Lead-Time Demand: {ltd} units | Forecast: {daily_d:.1f}/day",
                "Proposed Action": f"Place priority emergency replenishment of {order_qty} units immediately. Inquire with {row['supplier_name']} for expedited freight."
            })

        # Rule 2: Normal Reorder Threshold Breached
        elif status == "REORDER NOW":
            recommendations.append({
                "Product": p_name,
                "Category": cat,
                "Priority": "HIGH",
                "Identified Issue": f"Reorder Threshold Breached: Stock ({cur}) dropped below ROP ({rop}).",
                "Calculations": f"Lead-Time Demand: {ltd} + Safety Stock: {row['safety_stock']} = ROP: {rop}",
                "Proposed Action": f"Issue purchase order for {order_qty} units (Est. Cost: ${row['estimated_order_cost']:,.2f}) respecting supplier MOQ of {row['min_order_qty']}."
            })

        # Rule 3: Excess Stock & Capital Drain
        elif status == "EXCESS STOCK" and excess > 0:
            action_desc = "Pause replenishment orders."
            if trend < -5:
                action_desc = "Demand is cooling down (-{:.1f}%). Bundle with complementary products or run mild promotional discount.".format(abs(trend))
            elif row["seasonal_indicator"] in ["Summer Peak", "Winter Peak", "Holiday Peak"]:
                action_desc = f"Seasonal item ({row['seasonal_indicator']}). Verify if off-season storage costs exceed markdown liquidation."
            else:
                action_desc = "Review procurement batch sizes and consider reallocating surplus units to partner stores."

            recommendations.append({
                "Product": p_name,
                "Category": cat,
                "Priority": "MEDIUM",
                "Identified Issue": f"Excess Capital Tied Up: {excess} surplus units holding ${ex_val:,.2f}.",
                "Calculations": f"Current Coverage: {days_rem:.0f} days vs Target: {row['target_coverage_stock'] / daily_d:.0f} days | Sales Trend: {trend:+.1f}%",
                "Proposed Action": action_desc
            })

        # Rule 4: Supplier Reliability Risk
        if rel < 0.90 and cur < (rop * 1.3):
            recommendations.append({
                "Product": p_name,
                "Category": cat,
                "Priority": "WARNING",
                "Identified Issue": f"Low Supplier Reliability ({rel*100:.0f}%) from {row['supplier_name']}.",
                "Calculations": f"Safety stock inflated to {row['safety_stock']} units to absorb shipment delays.",
                "Proposed Action": "Audit supplier on-time performance or establish secondary qualification for dual-sourcing."
            })

    return recommendations


# ==============================================================================
# 5. MAIN APPLICATION CONTROLLER & UI
# ==============================================================================
def main():
    # Header Banner
    st.title("📦 AI Inventory Intelligence & Optimization Dashboard")
    st.markdown(
        "A transparent, decision-theoretic inventory control system designed to **mitigate stockout vulnerabilities** in fast-moving lines and **eliminate working capital drag** from slow-moving inventory."
    )

    # --------------------------------------------------------------------------
    # SIDEBAR CONTROLS
    # --------------------------------------------------------------------------
    st.sidebar.header("⚙️ Policy & Filter Controls")
    
    # 1. Target inventory coverage slider
    target_coverage_days = st.sidebar.slider(
        "Target Inventory Coverage (Days)",
        min_value=15,
        max_value=90,
        value=30,
        step=5,
        help="The maximum desired inventory coverage in days before stock is classified as excess."
    )

    # 2. Review period slider
    review_period_days = st.sidebar.slider(
        "Review Period / Replenishment Cycle (Days)",
        min_value=7,
        max_value=30,
        value=14,
        step=1,
        help="The frequency in days between regular purchase order reviews."
    )

    # Load and calculate data
    raw_df = generate_inventory_data()
    df = calculate_inventory_metrics(raw_df, target_coverage_days, review_period_days)

    st.sidebar.markdown("---")

    # 3. Category Filter
    categories = ["All"] + sorted(df["category"].unique().tolist())
    selected_category = st.sidebar.selectbox("Filter by Category", categories)

    # 4. Status Filter
    statuses = ["All", "STOCKOUT RISK", "REORDER NOW", "HEALTHY", "EXCESS STOCK"]
    selected_status = st.sidebar.selectbox("Filter by Inventory Status", statuses)

    # 5. Refresh button
    if st.sidebar.button("🔄 Recalculate & Refresh Analysis", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    # Apply filters
    filtered_df = df.copy()
    if selected_category != "All":
        filtered_df = filtered_df[filtered_df["category"] == selected_category]
    if selected_status != "All":
        filtered_df = filtered_df[filtered_df["inventory_status"] == selected_status]

    # --------------------------------------------------------------------------
    # KPI METRIC CARDS
    # --------------------------------------------------------------------------
    total_products = len(df)
    stockout_count = (df["inventory_status"] == "STOCKOUT RISK").sum()
    reorder_count = (df["inventory_status"] == "REORDER NOW").sum()
    excess_count = (df["inventory_status"] == "EXCESS STOCK").sum()
    excess_value = df["excess_inventory_value"].sum()

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric(
        label="Total Products Monitored",
        value=f"{total_products}",
        delta=f"{len(filtered_df)} in current view" if len(filtered_df) != total_products else "All active"
    )
    kpi2.metric(
        label="Stockout Risk (Immediate)",
        value=f"{stockout_count}",
        delta=f"{stockout_count + reorder_count} Total Reorders Needed",
        delta_color="inverse"
    )
    kpi3.metric(
        label="Excess Inventory Items",
        value=f"{excess_count}",
        delta=f"{(excess_count / total_products) * 100:.0f}% of catalog",
        delta_color="off"
    )
    kpi4.metric(
        label="Capital Tied in Excess Stock",
        value=f"${excess_value:,.2f}",
        delta="Holding cost drain",
        delta_color="inverse"
    )

    # --------------------------------------------------------------------------
    # HIGH-PRIORITY RISK ALERT SUMMARY
    # --------------------------------------------------------------------------
    urgent_stockouts = df[df["inventory_status"] == "STOCKOUT RISK"]
    top_excess = df[df["inventory_status"] == "EXCESS STOCK"].sort_values(by="excess_inventory_value", ascending=False).head(3)

    with st.expander("🚨 **Executive Inventory Alert Summary**", expanded=True):
        alert_col1, alert_col2 = st.columns(2)
        with alert_col1:
            st.markdown("##### ⚠️ Critical Stockout Vulnerabilities")
            if not urgent_stockouts.empty:
                for _, r in urgent_stockouts.iterrows():
                    st.error(
                        f"**{r['product_name']}** ({r['category']}): "
                        f"Only **{r['current_inventory']} units** left ({r['days_of_stock_remaining']:.1f} days). "
                        f"Supplier lead time is **{r['lead_time_days']} days**! Urgent PO of **{r['suggested_order_qty']} units** required."
                    )
            else:
                st.success("No critical stockouts detected under current demand parameters.")

        with alert_col2:
            st.markdown("##### 💸 Top Capital Tied in Surplus Stock")
            if not top_excess.empty:
                for _, r in top_excess.iterrows():
                    st.warning(
                        f"**{r['product_name']}** ({r['category']}): "
                        f"**{r['excess_units']} excess units** (${r['excess_inventory_value']:,.2f} tied capital). "
                        f"Coverage: **{r['days_of_stock_remaining']:.0f} days** vs {target_coverage_days}d target. Recommendation: Pause procurement."
                    )
            else:
                st.success("No significant excess capital tie-up detected.")

    st.markdown("---")

    # --------------------------------------------------------------------------
    # DASHBOARD TABS
    # --------------------------------------------------------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Overview & Risk Matrix",
        "📈 Demand Analysis & Trends",
        "🛒 Reorder Planning & Optimization",
        "📦 Excess Inventory Analysis",
        "🧠 Rule-Based Decision Engine"
    ])

    # --------------------------------------------------------------------------
    # TAB 1: OVERVIEW & RISK MATRIX
    # --------------------------------------------------------------------------
    with tab1:
        st.subheader("Inventory Status Matrix")
        st.caption(f"Showing {len(filtered_df)} products matching selected filter criteria.")

        display_cols = [
            "product_id", "product_name", "category", "current_inventory",
            "forecast_daily_demand", "days_of_stock_remaining", "lead_time_days",
            "reorder_point", "inventory_status", "total_inventory_value"
        ]
        
        display_df = filtered_df[display_cols].copy()
        display_df.rename(columns={
            "product_id": "SKU",
            "product_name": "Product Name",
            "category": "Category",
            "current_inventory": "Current Stock",
            "forecast_daily_demand": "Daily Demand",
            "days_of_stock_remaining": "Days Remaining",
            "lead_time_days": "Lead Time (d)",
            "reorder_point": "Reorder Point",
            "inventory_status": "Status",
            "total_inventory_value": "Current Valuation ($)"
        }, inplace=True)

        # Apply Streamlit dataframe with formatting
        st.dataframe(
            display_df.style.map(
                lambda val: (
                    "background-color: #ffcccc; color: #990000; font-weight: bold;" if val == "STOCKOUT RISK" else
                    "background-color: #fff3cd; color: #856404; font-weight: bold;" if val == "REORDER NOW" else
                    "background-color: #d4edda; color: #155724;" if val == "HEALTHY" else
                    "background-color: #e2e3e5; color: #383d41;" if val == "EXCESS STOCK" else ""
                ),
                subset=["Status"]
            ).format({
                "Daily Demand": "{:.2f}",
                "Days Remaining": "{:.1f}",
                "Current Valuation ($)": "${:,.2f}"
            }),
            use_container_width=True,
            height=400
        )

    # --------------------------------------------------------------------------
    # TAB 2: DEMAND ANALYSIS & FORECASTING
    # --------------------------------------------------------------------------
    with tab2:
        st.subheader("Historical Demand & Moving Average Trend")
        
        # Product selector for trend inspection
        available_products = filtered_df["product_name"].tolist() if not filtered_df.empty else df["product_name"].tolist()
        selected_prod_name = st.selectbox("Select Product to Inspect Monthly Sales History:", available_products)
        
        prod_row = df[df["product_name"] == selected_prod_name].iloc[0]
        
        # 1. Monthly sales line chart
        months = [f"M{str(i).zfill(2)}" for i in range(1, 13)]
        sales_data = pd.DataFrame({
            "Month": months,
            "Units Sold": [prod_row[m] for m in months]
        }).set_index("Month")

        col_trend_left, col_trend_right = st.columns([2, 1])
        with col_trend_left:
            st.markdown(f"##### 12-Month Sales History: {selected_prod_name}")
            st.line_chart(sales_data, color="#1f77b4")
        
        with col_trend_right:
            st.markdown("##### Demand Statistics")
            st.markdown(f"- **Category:** {prod_row['category']}")
            st.markdown(f"- **Average Daily Demand:** `{prod_row['avg_daily_demand']:.2f} units/day`")
            st.markdown(f"- **Weighted Forecast Daily:** `{prod_row['forecast_daily_demand']:.2f} units/day`")
            st.markdown(f"- **Recent Sales Trend:** `{prod_row['demand_trend_pct']:+.1f}%`")
            st.markdown(f"- **Demand Variability (Daily σ):** `{prod_row['daily_demand_std']:.2f}`")
            st.markdown(f"- **Seasonal Pattern:** `{prod_row['seasonal_indicator']}`")
            st.info(
                "Near-term demand is estimated using a 4-month Weighted Moving Average (weights 10%, 20%, 30%, 40%) "
                "with non-overfitted seasonal calibration."
            )

        st.markdown("---")
        st.subheader("Cross-Product Daily Demand Comparison")
        bar_df = filtered_df[["product_name", "forecast_daily_demand"]].copy()
        bar_df = bar_df.sort_values(by="forecast_daily_demand", ascending=False).set_index("product_name")
        st.bar_chart(bar_df, color="#ff7f0e", height=320)

        st.markdown("---")
        st.subheader("Demand Velocity: Fast-Growing vs. Declining Products")
        col_grow, col_fall = st.columns(2)
        with col_grow:
            st.markdown("##### 🚀 Highest Growing Demand (Recent vs Prior Quarter)")
            grow_df = df.sort_values(by="demand_trend_pct", ascending=False).head(5)[
                ["product_name", "category", "demand_trend_pct", "forecast_daily_demand"]
            ]
            st.dataframe(grow_df.rename(columns={
                "product_name": "Product", "category": "Category",
                "demand_trend_pct": "Growth Trend (%)", "forecast_daily_demand": "Daily Demand"
            }), use_container_width=True)

        with col_fall:
            st.markdown("##### 📉 Steepest Declining Demand")
            decl_df = df.sort_values(by="demand_trend_pct", ascending=True).head(5)[
                ["product_name", "category", "demand_trend_pct", "forecast_daily_demand"]
            ]
            st.dataframe(decl_df.rename(columns={
                "product_name": "Product", "category": "Category",
                "demand_trend_pct": "Decline Trend (%)", "forecast_daily_demand": "Daily Demand"
            }), use_container_width=True)

    # --------------------------------------------------------------------------
    # TAB 3: REORDER PLANNING & OPTIMIZATION
    # --------------------------------------------------------------------------
    with tab3:
        st.subheader("Automated Replenishment Schedule")
        st.markdown(
            "Suggested order quantities are calculated to cover **Supplier Lead Time + Review Period ({review_period_days} days) + Safety Stock**, "
            "strictly bounded by **Supplier Minimum Order Quantities (MOQ)** and whole integer rounding."
        )

        reorder_candidates = filtered_df[filtered_df["suggested_order_qty"] > 0].copy()

        if not reorder_candidates.empty:
            total_po_cost = reorder_candidates["estimated_order_cost"].sum()
            st.info(f"💡 Total Recommended Purchase Orders: **{len(reorder_candidates)} items** | Total Estimated Capital Outlay: **${total_po_cost:,.2f}**")

            reorder_view = reorder_candidates[[
                "product_name", "supplier_name", "current_inventory",
                "forecast_daily_demand", "days_of_stock_remaining", "reorder_point",
                "min_order_qty", "suggested_order_qty", "estimated_order_cost", "reorder_reason"
            ]].copy()

            reorder_view.rename(columns={
                "product_name": "Product",
                "supplier_name": "Supplier",
                "current_inventory": "Current Stock",
                "forecast_daily_demand": "Forecast Daily Demand",
                "days_of_stock_remaining": "Days Remaining",
                "reorder_point": "Reorder Point",
                "min_order_qty": "MOQ",
                "suggested_order_qty": "Suggested Order Qty",
                "estimated_order_cost": "Estimated Cost ($)",
                "reorder_reason": "Rationale"
            }, inplace=True)

            st.dataframe(
                reorder_view.style.format({
                    "Forecast Daily Demand": "{:.2f}",
                    "Days Remaining": "{:.1f}",
                    "Estimated Cost ($)": "${:,.2f}"
                }),
                use_container_width=True
            )
        else:
            st.success("No items currently require replenishment under current filter criteria.")

    # --------------------------------------------------------------------------
    # TAB 4: EXCESS INVENTORY ANALYSIS
    # --------------------------------------------------------------------------
    with tab4:
        st.subheader("Excess & Slow-Moving Stock Diagnostic")
        st.markdown(
            f"Products whose current stock substantially exceeds demand over the target coverage period (**{target_coverage_days} days**). "
            "Proposes balanced disposition strategies rather than default knee-jerk discounting."
        )

        excess_items = filtered_df[filtered_df["excess_units"] > 0].copy()

        if not excess_items.empty:
            excess_view = excess_items[[
                "product_name", "category", "current_inventory", "forecast_daily_demand",
                "target_coverage_stock", "excess_units", "unit_cost", "excess_inventory_value",
                "demand_trend_pct", "seasonal_indicator"
            ]].copy()

            # Categorize tailored disposition action
            def assign_excess_action(row):
                if row["demand_trend_pct"] < -10:
                    return "Promotional Bundle / Markdowns (Declining Velocity)"
                elif row["seasonal_indicator"] != "Stable / None":
                    return "Pre-Season Hold / Targeted Seasonal Campaign"
                elif row["unit_cost"] > 15:
                    return "Inter-Branch Stock Transfer (High Value Asset)"
                else:
                    return "Pause Replenishment Orders & Adjust Procurement Cycle"

            excess_view["Recommended Disposition"] = excess_view.apply(assign_excess_action, axis=1)

            excess_view.rename(columns={
                "product_name": "Product",
                "category": "Category",
                "current_inventory": "Stock on Hand",
                "forecast_daily_demand": "Daily Demand",
                "target_coverage_stock": f"Target Stock ({target_coverage_days}d)",
                "excess_units": "Surplus Units",
                "unit_cost": "Unit Cost ($)",
                "excess_inventory_value": "Surplus Value ($)",
                "demand_trend_pct": "Recent Trend (%)",
                "seasonal_indicator": "Seasonality"
            }, inplace=True)

            st.dataframe(
                excess_view.style.format({
                    "Daily Demand": "{:.2f}",
                    "Unit Cost ($)": "${:.2f}",
                    "Surplus Value ($)": "${:,.2f}",
                    "Recent Trend (%)": "{:+.1f}%"
                }),
                use_container_width=True
            )
        else:
            st.success(f"No products exceed the {target_coverage_days}-day target inventory threshold.")

    # --------------------------------------------------------------------------
    # TAB 5: TRANSPARENT RULE-BASED RECOMMENDATION ENGINE
    # --------------------------------------------------------------------------
    with tab5:
        st.subheader("Transparent, Explainable Inventory Decision Log")
        st.markdown(
            "Every recommendation provides full traceability from the mathematical calculations to the executive decision. "
            "No black-box models or untraceable outputs."
        )

        insights = generate_prescriptive_insights(filtered_df)

        if insights:
            rec_df = pd.DataFrame(insights)
            st.dataframe(
                rec_df.style.map(
                    lambda p: (
                        "background-color: #f8d7da; color: #721c24; font-weight: bold;" if p == "CRITICAL" else
                        "background-color: #fff3cd; color: #856404; font-weight: bold;" if p == "HIGH" else
                        "background-color: #d1ecf1; color: #0c5460; font-weight: bold;" if p == "WARNING" else
                        "background-color: #e2e3e5; color: #383d41;" if p == "MEDIUM" else ""
                    ),
                    subset=["Priority"]
                ),
                use_container_width=True,
                height=500
            )
        else:
            st.info("No rule triggers fired for the selected filter combination.")

    # --------------------------------------------------------------------------
    # FOOTER & ACADEMIC NOTES
    # --------------------------------------------------------------------------
    st.markdown("---")
    st.caption(
        "🎓 **Academic Project Note:** Deterministic simulation based on classical inventory theory "
        "(Safety Stock with 95% service level, Lead-Time Demand, Continuous Review ROP, and Weighted Moving Average forecasting). "
        "Built strictly with Streamlit, Pandas, and NumPy. No external APIs or paid LLMs required."
    )


if __name__ == "__main__":
    main()
