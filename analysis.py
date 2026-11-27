
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load cleaned dataset
df = pd.read_csv("outputs/nassau_cleaned.csv")

# Convert Order Date
df["Order Date"] = pd.to_datetime(
    df["Order Date"],
    format="%Y-%m-%d",
    errors="coerce"
)

df["Ship Date"] = pd.to_datetime(
    df["Ship Date"],
    format="%Y-%m-%d",
    errors="coerce"
)

os.makedirs("outputs/charts", exist_ok=True)

print("\n========== EXPLORATORY DATA ANALYSIS ==========")

# 1. Overall Business Performance
total_sales = df["Sales"].sum()
total_profit = df["Gross Profit"].sum()
total_cost = df["Cost"].sum()
total_units = df["Units"].sum()

gross_margin = (total_profit / total_sales) * 100

print("\n1. OVERALL BUSINESS PERFORMANCE")
print("Total Sales:", round(total_sales, 2))
print("Total Profit:", round(total_profit, 2))
print("Total Cost:", round(total_cost, 2))
print("Total Units:", total_units)
print("Gross Margin:", round(gross_margin, 2), "%")

# 2. Product-wise Analysis
product_analysis = df.groupby("Product Name").agg(
    Sales=("Sales", "sum"),
    Profit=("Gross Profit", "sum"),
    Cost=("Cost", "sum"),
    Units=("Units", "sum")
)

product_analysis["Margin (%)"] = (
    product_analysis["Profit"] /
    product_analysis["Sales"] * 100
)

product_analysis = product_analysis.sort_values(
    by="Profit", ascending=False
)

print("\n2. PRODUCT-WISE PROFIT ANALYSIS")
print(product_analysis.round(2))

# Top 10 products by profit
top_products = product_analysis.head(10)

plt.figure(figsize=(12, 6))
top_products["Profit"].plot(kind="bar")
plt.title("Top 10 Products by Gross Profit")
plt.xlabel("Product Name")
plt.ylabel("Gross Profit")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("outputs/charts/top_products.png")
plt.close()

# 3. Division-wise Analysis
division_analysis = df.groupby("Division").agg(
    Sales=("Sales", "sum"),
    Profit=("Gross Profit", "sum"),
    Cost=("Cost", "sum"),
    Units=("Units", "sum")
)

division_analysis["Margin (%)"] = (
    division_analysis["Profit"] /
    division_analysis["Sales"] * 100
)

print("\n3. DIVISION-WISE ANALYSIS")
print(division_analysis.round(2))

# Revenue vs Profit chart
division_analysis[["Sales", "Profit"]].plot(
    kind="bar", figsize=(9, 5)
)
plt.title("Division-wise Revenue vs Profit")
plt.xlabel("Division")
plt.ylabel("Amount")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("outputs/charts/division_performance.png")
plt.close()

# 4. Monthly Sales Analysis
df["Month"] = df["Order Date"].dt.to_period("M")

monthly_analysis = df.groupby("Month").agg(
    Sales=("Sales", "sum"),
    Profit=("Gross Profit", "sum")
)

monthly_analysis.index = monthly_analysis.index.astype(str)

print("\n4. MONTHLY SALES ANALYSIS")
print(monthly_analysis.round(2))

plt.figure(figsize=(13, 6))
plt.plot(
    monthly_analysis.index,
    monthly_analysis["Sales"],
    marker="o"
)
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("outputs/charts/monthly_sales.png")
plt.close()

# 5. Region-wise Analysis
region_analysis = df.groupby("Region").agg(
    Sales=("Sales", "sum"),
    Profit=("Gross Profit", "sum")
)

print("\n5. REGION-WISE ANALYSIS")
print(region_analysis.round(2))

region_analysis[["Sales", "Profit"]].plot(
    kind="bar", figsize=(9, 5)
)
plt.title("Region-wise Sales and Profit")
plt.xlabel("Region")
plt.ylabel("Amount")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("outputs/charts/region_performance.png")
plt.close()

print("\n========== EDA COMPLETED ==========")
print("Charts saved in outputs/charts folder!")

# ==========================================
# PHASE 4: PRODUCT PROFITABILITY ANALYSIS
# ==========================================

print("\n========== PRODUCT PROFITABILITY ANALYSIS ==========")

# 1. Aggregate product metrics
product_metrics = df.groupby(
    ["Product Name", "Division"]
).agg(
    Sales=("Sales", "sum"),
    Gross_Profit=("Gross Profit", "sum"),
    Cost=("Cost", "sum"),
    Units=("Units", "sum")
).reset_index()

# 2. Calculate profitability KPIs
product_metrics["Gross_Margin"] = (
    product_metrics["Gross_Profit"] /
    product_metrics["Sales"] * 100
)

product_metrics["Profit_Per_Unit"] = (
    product_metrics["Gross_Profit"] /
    product_metrics["Units"]
)

product_metrics["Revenue_Contribution"] = (
    product_metrics["Sales"] / total_sales * 100
)

product_metrics["Profit_Contribution"] = (
    product_metrics["Gross_Profit"] / total_profit * 100
)

# 3. Rank products
product_metrics["Profit_Rank"] = (
    product_metrics["Gross_Profit"]
    .rank(method="min", ascending=False)
    .astype(int)
)

product_metrics["Margin_Rank"] = (
    product_metrics["Gross_Margin"]
    .rank(method="min", ascending=False)
    .astype(int)
)

# 4. Identify margin risks
product_metrics["Margin_Status"] = np.where(
    product_metrics["Gross_Margin"] < gross_margin,
    "Below Overall Margin",
    "Above Overall Margin"
)

# 5. Product performance categories
median_sales = product_metrics["Sales"].median()

product_metrics["Performance_Category"] = np.select(
    [
        (product_metrics["Sales"] >= median_sales) &
        (product_metrics["Gross_Margin"] >= gross_margin),

        (product_metrics["Sales"] >= median_sales) &
        (product_metrics["Gross_Margin"] < gross_margin),

        (product_metrics["Sales"] < median_sales) &
        (product_metrics["Gross_Margin"] >= gross_margin)
    ],
    [
        "High Sales - High Margin",
        "High Sales - Low Margin",
        "Low Sales - High Margin"
    ],
    default="Low Sales - Low Margin"
)

# 6. Sort by gross profit
product_metrics = product_metrics.sort_values(
    by="Gross_Profit",
    ascending=False
)

# 7. Display complete profitability table
display_columns = [
    "Product Name",
    "Division",
    "Sales",
    "Gross_Profit",
    "Gross_Margin",
    "Profit_Per_Unit",
    "Revenue_Contribution",
    "Profit_Contribution",
    "Profit_Rank",
    "Margin_Status",
    "Performance_Category"
]

print("\nPRODUCT PROFITABILITY TABLE")
print(
    product_metrics[display_columns].round(2).to_string(
        index=False
    )
)

# 8. Top 5 products by profit
print("\nTOP 5 PROFITABLE PRODUCTS")
print(
    product_metrics.nlargest(5, "Gross_Profit")[
        ["Product Name", "Gross_Profit", "Gross_Margin"]
    ].round(2).to_string(index=False)
)

# 9. Lowest 5 products by margin
print("\nLOWEST 5 PRODUCTS BY MARGIN")
print(
    product_metrics.nsmallest(5, "Gross_Margin")[
        ["Product Name", "Gross_Margin", "Gross_Profit"]
    ].round(2).to_string(index=False)
)

# 10. Pareto analysis - Profit contribution
pareto = product_metrics.sort_values(
    by="Gross_Profit",
    ascending=False
).copy()

pareto["Cumulative_Profit"] = (
    pareto["Gross_Profit"].cumsum()
)

pareto["Cumulative_Profit_Percentage"] = (
    pareto["Cumulative_Profit"] / total_profit * 100
)

pareto["Cumulative_Revenue"] = pareto["Sales"].cumsum()

pareto["Cumulative_Revenue_Percentage"] = (
    pareto["Cumulative_Revenue"] / total_sales * 100
)

print("\nPROFIT PARETO ANALYSIS")
print(
    pareto[
        ["Product Name", "Cumulative_Profit_Percentage"]
    ].round(2).to_string(index=False)
)

products_for_80_profit = (
    pareto["Cumulative_Profit_Percentage"] < 80
).sum() + 1

products_for_80_revenue = (
    pareto.sort_values("Sales", ascending=False)[
        "Sales"
    ].cumsum().lt(total_sales * 0.80).sum() + 1
)

print("\nProducts needed to reach approximately 80% profit:",
      products_for_80_profit)

print("Products needed to reach approximately 80% revenue:",
      products_for_80_revenue)

# 11. Save profitability results
product_metrics.to_csv(
    "outputs/product_profitability.csv",
    index=False
)

pareto.to_csv(
    "outputs/profit_pareto_analysis.csv",
    index=False
)

# 12. Profit contribution chart
plt.figure(figsize=(12, 6))

plt.bar(
    product_metrics["Product Name"],
    product_metrics["Profit_Contribution"]
)

plt.title("Product-wise Profit Contribution")
plt.xlabel("Product")
plt.ylabel("Profit Contribution (%)")
plt.xticks(rotation=75, ha="right")
plt.tight_layout()

plt.savefig(
    "outputs/charts/profit_contribution.png"
)
plt.close()

# 13. Cost vs Sales scatter plot
plt.figure(figsize=(9, 6))

plt.scatter(
    product_metrics["Sales"],
    product_metrics["Cost"]
)

plt.title("Product Cost vs Sales")
plt.xlabel("Total Sales")
plt.ylabel("Total Cost")
plt.tight_layout()

plt.savefig(
    "outputs/charts/cost_vs_sales.png"
)
plt.close()

# 14. Save Pareto chart
plt.figure(figsize=(12, 6))

plt.bar(
    pareto["Product Name"],
    pareto["Gross_Profit"]
)

plt.plot(
    range(len(pareto)),
    pareto["Cumulative_Profit_Percentage"],
    marker="o",
    color="red"
)

plt.axhline(
    y=80,
    linestyle="--",
    color="green",
    label="80% Profit Threshold"
)

plt.title("Product Profit Pareto Analysis")
plt.xlabel("Products")
plt.ylabel("Gross Profit / Cumulative Profit %")
plt.xticks(rotation=75, ha="right")
plt.legend()
plt.tight_layout()

plt.savefig(
    "outputs/charts/profit_pareto.png"
)
plt.close()

print("\n========== PHASE 4 COMPLETED ==========")
print("Profitability files saved successfully!")

# ==========================================
# PHASE 5: FACTORY-LEVEL PROFITABILITY
# ==========================================

print("\n========== FACTORY-LEVEL ANALYSIS ==========")

# 1. Factory coordinates
factory_coordinates = {
    "Lot's O' Nuts": {
        "Latitude": 32.881893,
        "Longitude": -111.768036
    },
    "Wicked Choccy's": {
        "Latitude": 32.076176,
        "Longitude": -81.088371
    },
    "Sugar Shack": {
        "Latitude": 48.11914,
        "Longitude": -96.18115
    },
    "Secret Factory": {
        "Latitude": 41.446333,
        "Longitude": -90.565487
    },
    "The Other Factory": {
        "Latitude": 35.1175,
        "Longitude": -89.971107
    }
}

# 2. Product-to-factory mapping
product_factory = {
    "Wonka Bar - Nutty Crunch Surprise": "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows": "Lot's O' Nuts",
    "Wonka Bar - Scrumdiddlyumptious": "Lot's O' Nuts",
    "Wonka Bar - Milk Chocolate": "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel": "Wicked Choccy's",
    "Laffy Taffy": "Sugar Shack",
    "SweeTARTS": "Sugar Shack",
    "Nerds": "Sugar Shack",
    "Fun Dip": "Sugar Shack",
    "Fizzy Lifting Drinks": "Sugar Shack",
    "Everlasting Gobstopper": "Secret Factory",
    "Lickable Wallpaper": "Secret Factory",
    "Wonka Gum": "Secret Factory",
    "Hair Toffee": "The Other Factory",
    "Kazookles": "The Other Factory"
}

# Normalize product names to avoid differences in spaces,
# hyphens, and capital letters.
def normalize_product_name(name):
    return "".join(
        character.lower()
        for character in str(name)
        if character.isalnum()
    )

normalized_mapping = {
    normalize_product_name(product): factory
    for product, factory in product_factory.items()
}

# 3. Add factory to the original dataset
df["Factory"] = df["Product Name"].apply(
    lambda name: normalized_mapping.get(
        normalize_product_name(name),
        "Unmapped"
    )
)

# Add factory coordinates
df["Factory_Latitude"] = df["Factory"].map(
    {
        factory: details["Latitude"]
        for factory, details in factory_coordinates.items()
    }
)

df["Factory_Longitude"] = df["Factory"].map(
    {
        factory: details["Longitude"]
        for factory, details in factory_coordinates.items()
    }
)

# Check unmatched products
unmapped_products = df.loc[
    df["Factory"] == "Unmapped",
    "Product Name"
].unique()

print("\nUNMAPPED PRODUCTS:")
print(unmapped_products)

# 4. Factory-wise performance
factory_analysis = df.groupby("Factory").agg(
    Sales=("Sales", "sum"),
    Gross_Profit=("Gross Profit", "sum"),
    Cost=("Cost", "sum"),
    Units=("Units", "sum"),
    Orders=("Order ID", "nunique"),
    Products=("Product Name", "nunique")
).reset_index()

factory_analysis["Gross_Margin"] = (
    factory_analysis["Gross_Profit"] /
    factory_analysis["Sales"] * 100
)

factory_analysis["Profit_Contribution"] = (
    factory_analysis["Gross_Profit"] /
    total_profit * 100
)

factory_analysis = factory_analysis.sort_values(
    "Gross_Profit",
    ascending=False
)

print("\nFACTORY-WISE PERFORMANCE")
print(factory_analysis.round(2).to_string(index=False))

# 5. Save factory performance
factory_analysis.to_csv(
    "outputs/factory_performance.csv",
    index=False
)

# 6. Factory-wise profit chart
plt.figure(figsize=(10, 6))

plt.bar(
    factory_analysis["Factory"],
    factory_analysis["Gross_Profit"]
)

plt.title("Factory-wise Gross Profit")
plt.xlabel("Factory")
plt.ylabel("Gross Profit")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

plt.savefig(
    "outputs/charts/factory_profit.png"
)
plt.close()

# 7. Factory-wise sales chart
plt.figure(figsize=(10, 6))

plt.bar(
    factory_analysis["Factory"],
    factory_analysis["Sales"]
)

plt.title("Factory-wise Revenue")
plt.xlabel("Factory")
plt.ylabel("Sales")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

plt.savefig(
    "outputs/charts/factory_sales.png"
)
plt.close()

# 8. Save dataset with factory mapping
df.to_csv(
    "outputs/nassau_factory_mapped.csv",
    index=False
)

print("\nFactory mapping saved successfully.")


# ==========================================
# PHASE 6: STATE AND REGION ORDER ANALYSIS
# ==========================================

print("\n========== GEOGRAPHICAL ORDER ANALYSIS ==========")

# 1. Region-level order analysis
region_shipping = df.groupby("Region").agg(
    Total_Orders=("Order ID", "nunique"),
    Total_Sales=("Sales", "sum"),
    Total_Profit=("Gross Profit", "sum"),
    Total_Units=("Units", "sum"),
    States=("State/Province", "nunique")
).reset_index()

print("\nREGION-WISE ORDER ANALYSIS")
print(region_shipping.round(2).to_string(index=False))

# 2. State-level order analysis
state_shipping = df.groupby(
    ["State/Province", "Region"]
).agg(
    Total_Orders=("Order ID", "nunique"),
    Total_Sales=("Sales", "sum"),
    Total_Profit=("Gross Profit", "sum"),
    Total_Units=("Units", "sum")
).reset_index()

state_shipping = state_shipping.sort_values(
    "Total_Orders",
    ascending=False
)

print("\nTOP 10 STATES BY ORDER VOLUME")
print(state_shipping.head(10).round(2).to_string(index=False))

# 3. Save geographical analysis
region_shipping.to_csv(
    "outputs/region_shipping_analysis.csv",
    index=False
)

state_shipping.to_csv(
    "outputs/state_shipping_analysis.csv",
    index=False
)

# 4. Top states by order volume
top_states = state_shipping.head(10)

plt.figure(figsize=(12, 6))

plt.bar(
    top_states["State/Province"],
    top_states["Total_Orders"]
)

plt.title("Top 10 States by Order Volume")
plt.xlabel("State / Province")
plt.ylabel("Number of Orders")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig(
    "outputs/charts/top_states_orders.png"
)
plt.close()

# 5. Region-wise order volume
plt.figure(figsize=(9, 5))

plt.bar(
    region_shipping["Region"],
    region_shipping["Total_Orders"]
)

plt.title("Region-wise Order Volume")
plt.xlabel("Region")
plt.ylabel("Number of Orders")
plt.tight_layout()

plt.savefig(
    "outputs/charts/region_order_volume.png"
)
plt.close()

print("\nGeographical order analysis saved successfully.")


# ==========================================
# PHASE 7: PRODUCT MARGIN VOLATILITY
# ==========================================

print("\n========== MARGIN VOLATILITY ANALYSIS ==========")

# Calculate monthly product margins
monthly_product = df.groupby(
    [
        "Product Name",
        pd.Grouper(key="Order Date", freq="MS")
    ]
).agg(
    Monthly_Sales=("Sales", "sum"),
    Monthly_Profit=("Gross Profit", "sum")
).reset_index()

monthly_product["Monthly_Margin"] = np.where(
    monthly_product["Monthly_Sales"] > 0,
    monthly_product["Monthly_Profit"] /
    monthly_product["Monthly_Sales"] * 100,
    np.nan
)

margin_volatility = monthly_product.groupby(
    "Product Name"
)["Monthly_Margin"].std().reset_index()

margin_volatility = margin_volatility.rename(
    columns={
        "Monthly_Margin": "Margin_Volatility"
    }
)

margin_volatility["Margin_Volatility"] = (
    margin_volatility["Margin_Volatility"].fillna(0)
)

print("\nPRODUCT MARGIN VOLATILITY")
print(
    margin_volatility.sort_values(
        "Margin_Volatility",
        ascending=False
    ).round(2).to_string(index=False)
)

margin_volatility.to_csv(
    "outputs/margin_volatility.csv",
    index=False
)

print("\n========== ALL NEW ANALYSIS PHASES COMPLETED ==========")
print("Factory, geographical and volatility outputs saved.")
