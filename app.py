import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Nassau Candy Profitability Dashboard",
    page_icon="🍫",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ==================================================
# 2. FACTORY COORDINATES
# ==================================================

FACTORY_COORDINATES = {
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


# ==================================================
# 3. PRODUCT-TO-FACTORY MAPPING
# ==================================================

PRODUCT_FACTORY = {
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


def normalize_product_name(name):
    return "".join(
        ch.lower()
        for ch in str(name)
        if ch.isalnum()
    )


NORMALIZED_PRODUCT_FACTORY = {
    normalize_product_name(product): factory
    for product, factory in PRODUCT_FACTORY.items()
}


# ==================================================
# 4. LOAD DATA
# ==================================================

@st.cache_data
def load_data():

    file_path = BASE_DIR / "nassau_cleaned.csv"

    if not file_path.exists():
        raise FileNotFoundError(
            "outputs/nassau_cleaned.csv not found. "
            "Run analysis.py first."
        )

    data = pd.read_csv(file_path)

    required_columns = [
        "Order Date",
        "Product Name",
        "Division",
        "Sales",
        "Gross Profit",
        "Cost",
        "Units",
        "Order ID",
        "State/Province",
        "Region"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    data["Order Date"] = pd.to_datetime(
        data["Order Date"],
        errors="coerce"
    )

    for col in ["Sales", "Gross Profit", "Cost", "Units"]:
        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        ).fillna(0)

    data["Factory"] = data["Product Name"].apply(
        lambda name: NORMALIZED_PRODUCT_FACTORY.get(
            normalize_product_name(name),
            "Unmapped"
        )
    )

    data["Factory_Latitude"] = data["Factory"].map(
        {
            name: details["Latitude"]
            for name, details in FACTORY_COORDINATES.items()
        }
    )

    data["Factory_Longitude"] = data["Factory"].map(
        {
            name: details["Longitude"]
            for name, details in FACTORY_COORDINATES.items()
        }
    )

    return data


try:
    df = load_data()

except Exception as error:
    st.error(f"Unable to load data: {error}")
    st.stop()


if df["Order Date"].dropna().empty:
    st.error("No valid order dates found.")
    st.stop()


# ==================================================
# 5. DASHBOARD TITLE
# ==================================================

st.title("🍫 Nassau Candy Distributor")

st.subheader(
    "Product Line Profitability & Margin Performance Analysis"
)

st.write(
    "Explore product profitability, division performance, "
    "cost structure, profit concentration, factory performance "
    "and geographical order patterns."
)


# ==================================================
# 6. SIDEBAR FILTERS
# ==================================================

st.sidebar.header("Dashboard Filters")

min_date = df["Order Date"].min().date()
max_date = df["Order Date"].max().date()

date_range = st.sidebar.date_input(
    "Select Order Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

divisions = sorted(
    df["Division"].dropna().unique()
)

selected_divisions = st.sidebar.multiselect(
    "Select Division",
    divisions,
    default=divisions
)

margin_threshold = st.sidebar.slider(
    "Minimum Gross Margin (%)",
    min_value=0,
    max_value=100,
    value=0
)

product_search = st.sidebar.text_input(
    "Search Product"
)


# ==================================================
# 7. APPLY DATE AND DIVISION FILTERS
# ==================================================

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range

elif isinstance(date_range, (tuple, list)) and len(date_range) == 1:
    start_date = end_date = date_range[0]

else:
    start_date, end_date = min_date, max_date


filtered_df = df[
    df["Order Date"].dt.date.between(
        start_date,
        end_date
    )
    &
    df["Division"].isin(selected_divisions)
].copy()


if filtered_df.empty:
    st.warning(
        "No records match the selected filters."
    )
    st.stop()


# ==================================================
# 8. PRODUCT PROFITABILITY METRICS
# ==================================================

product_metrics = (
    filtered_df.groupby(
        ["Product Name", "Division"],
        as_index=False
    )
    .agg(
        Sales=("Sales", "sum"),
        Gross_Profit=("Gross Profit", "sum"),
        Cost=("Cost", "sum"),
        Units=("Units", "sum")
    )
)

product_metrics["Gross_Margin"] = np.where(
    product_metrics["Sales"] > 0,
    product_metrics["Gross_Profit"]
    / product_metrics["Sales"] * 100,
    0
)

product_metrics["Profit_Per_Unit"] = np.where(
    product_metrics["Units"] > 0,
    product_metrics["Gross_Profit"]
    / product_metrics["Units"],
    0
)

total_sales = product_metrics["Sales"].sum()
total_profit = product_metrics["Gross_Profit"].sum()

product_metrics["Revenue_Contribution"] = np.where(
    total_sales > 0,
    product_metrics["Sales"] / total_sales * 100,
    0
)

product_metrics["Profit_Contribution"] = np.where(
    total_profit > 0,
    product_metrics["Gross_Profit"] / total_profit * 100,
    0
)


# ==================================================
# 9. MARGIN VOLATILITY
# ==================================================

monthly_product = filtered_df.copy()

monthly_product["Month"] = (
    monthly_product["Order Date"].dt.to_period("M")
)

monthly_product = (
    monthly_product.groupby(
        ["Product Name", "Month"],
        as_index=False
    )
    .agg(
        Monthly_Sales=("Sales", "sum"),
        Monthly_Profit=("Gross Profit", "sum")
    )
)

monthly_product["Monthly_Margin"] = np.where(
    monthly_product["Monthly_Sales"] > 0,
    monthly_product["Monthly_Profit"]
    / monthly_product["Monthly_Sales"] * 100,
    np.nan
)

volatility = (
    monthly_product.groupby("Product Name")[
        "Monthly_Margin"
    ]
    .std()
    .fillna(0)
    .reset_index(name="Margin_Volatility")
)

product_metrics = product_metrics.merge(
    volatility,
    on="Product Name",
    how="left"
)

product_metrics["Margin_Volatility"] = (
    product_metrics["Margin_Volatility"].fillna(0)
)


# ==================================================
# 10. MARGIN RISK CLASSIFICATION
# ==================================================

product_metrics["Margin_Risk"] = pd.cut(
    product_metrics["Gross_Margin"],
    bins=[-np.inf, 40, 60, np.inf],
    labels=[
        "High Risk",
        "Medium Risk",
        "Healthy"
    ],
    right=False
).astype(str)


median_sales = product_metrics["Sales"].median()

overall_margin = (
    total_profit / total_sales * 100
    if total_sales > 0 else 0
)

product_metrics["Performance_Category"] = np.select(
    [
        (product_metrics["Sales"] >= median_sales)
        & (product_metrics["Gross_Margin"] >= overall_margin),

        (product_metrics["Sales"] >= median_sales)
        & (product_metrics["Gross_Margin"] < overall_margin),

        (product_metrics["Sales"] < median_sales)
        & (product_metrics["Gross_Margin"] >= overall_margin)
    ],
    [
        "High Sales - High Margin",
        "High Sales - Low Margin",
        "Low Sales - High Margin"
    ],
    default="Low Sales - Low Margin"
)


product_metrics = product_metrics.merge(
    filtered_df[
        ["Product Name", "Factory"]
    ].drop_duplicates(),
    on="Product Name",
    how="left"
)


# ==================================================
# 11. PRODUCT FILTERS
# ==================================================

filtered_products = product_metrics[
    product_metrics["Gross_Margin"] >= margin_threshold
].copy()

if product_search:
    filtered_products = filtered_products[
        filtered_products["Product Name"].str.contains(
            product_search,
            case=False,
            na=False
        )
    ]


# ==================================================
# 12. BUSINESS PERFORMANCE OVERVIEW
# ==================================================

st.markdown("## 1. Business Performance Overview")

total_sales = filtered_df["Sales"].sum()
total_profit = filtered_df["Gross Profit"].sum()
total_cost = filtered_df["Cost"].sum()
total_units = filtered_df["Units"].sum()

overall_margin = (
    total_profit / total_sales * 100
    if total_sales > 0 else 0
)

monthly_totals = (
    filtered_df.groupby(
        filtered_df["Order Date"].dt.to_period("M")
    )
    .agg(
        Sales=("Sales", "sum"),
        Profit=("Gross Profit", "sum")
    )
)

monthly_totals["Margin"] = np.where(
    monthly_totals["Sales"] > 0,
    monthly_totals["Profit"]
    / monthly_totals["Sales"] * 100,
    np.nan
)

margin_volatility = (
    monthly_totals["Margin"].std()
    if monthly_totals["Margin"].count() > 1
    else 0
)

if pd.isna(margin_volatility):
    margin_volatility = 0

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Total Revenue", f"${total_sales:,.2f}")
c2.metric("Gross Profit", f"${total_profit:,.2f}")
c3.metric("Total Cost", f"${total_cost:,.2f}")
c4.metric("Gross Margin", f"{overall_margin:.2f}%")
c5.metric("Margin Volatility", f"{margin_volatility:.2f}%")

st.caption(f"Total Units Sold: {total_units:,.0f}")

st.caption(
    "Business KPIs respond to date and division filters. "
    "Product search and margin threshold affect product-level views only."
)

st.divider()


# ==================================================
# 13. PRODUCT PROFITABILITY ANALYSIS
# ==================================================

st.markdown("## 2. Product Profitability Analysis")

if not filtered_products.empty:

    col1, col2 = st.columns(2)

    with col1:

        top_products = filtered_products.nlargest(
            10,
            "Gross_Profit"
        )

        fig = px.bar(
            top_products.sort_values("Gross_Profit"),
            x="Gross_Profit",
            y="Product Name",
            orientation="h",
            title="Top Products by Gross Profit"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.bar(
            filtered_products.sort_values("Gross_Margin"),
            x="Gross_Margin",
            y="Product Name",
            orientation="h",
            color="Margin_Risk",
            title="Product Gross Margin and Risk"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("### Product Performance Table")

    display_columns = [
        "Product Name",
        "Division",
        "Factory",
        "Sales",
        "Gross_Profit",
        "Cost",
        "Units",
        "Gross_Margin",
        "Profit_Per_Unit",
        "Revenue_Contribution",
        "Profit_Contribution",
        "Margin_Volatility",
        "Margin_Risk",
        "Performance_Category"
    ]

    st.dataframe(
        filtered_products[display_columns]
        .sort_values(
            "Gross_Profit",
            ascending=False
        )
        .round(2),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Product Profit Contribution")

    contribution_data = filtered_products.nlargest(
        10,
        "Profit_Contribution"
    )

    fig = px.bar(
        contribution_data.sort_values(
            "Profit_Contribution"
        ),
        x="Profit_Contribution",
        y="Product Name",
        orientation="h",
        title="Top Products by Profit Contribution (%)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("### Margin Risk Flags")

    risk_products = filtered_products[
        filtered_products["Margin_Risk"] != "Healthy"
    ].sort_values("Gross_Margin")

    if not risk_products.empty:

        st.dataframe(
            risk_products[
                [
                    "Product Name",
                    "Factory",
                    "Sales",
                    "Gross_Profit",
                    "Gross_Margin",
                    "Margin_Risk"
                ]
            ].round(2),
            use_container_width=True,
            hide_index=True
        )

    else:
        st.success(
            "No low-margin products in the selected product view."
        )

else:
    st.warning(
        "No products match the selected margin threshold and search."
    )

st.caption(
    "Margin risk bands are provisional screening thresholds: "
    "below 40% = High Risk; 40% to below 60% = Medium Risk; "
    "60% or above = Healthy."
)

st.divider()


# ==================================================
# 14. DIVISION PERFORMANCE
# ==================================================

st.markdown("## 3. Division Performance")

division_data = (
    filtered_df.groupby(
        "Division",
        as_index=False
    )
    .agg(
        Sales=("Sales", "sum"),
        Gross_Profit=("Gross Profit", "sum"),
        Cost=("Cost", "sum"),
        Units=("Units", "sum")
    )
)

division_data["Gross_Margin"] = np.where(
    division_data["Sales"] > 0,
    division_data["Gross_Profit"]
    / division_data["Sales"] * 100,
    0
)

division_data["Profit_Contribution"] = np.where(
    total_profit > 0,
    division_data["Gross_Profit"]
    / total_profit * 100,
    0
)

if not division_data.empty:

    col1, col2 = st.columns(2)

    with col1:

        fig = px.bar(
            division_data,
            x="Division",
            y=["Sales", "Gross_Profit"],
            barmode="group",
            title="Revenue vs Gross Profit by Division"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.bar(
            division_data,
            x="Division",
            y="Gross_Margin",
            color="Division",
            title="Gross Margin by Division"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.dataframe(
        division_data.round(2),
        use_container_width=True,
        hide_index=True
    )

st.divider()


# ==================================================
# 15. COST VS SALES DIAGNOSTICS
# ==================================================

st.markdown("## 4. Cost vs Sales Diagnostics")

if not filtered_products.empty:

    fig = px.scatter(
        filtered_products,
        x="Sales",
        y="Cost",
        size="Gross_Profit",
        color="Margin_Risk",
        hover_name="Product Name",
        hover_data=[
            "Gross_Margin",
            "Gross_Profit",
            "Profit_Per_Unit",
            "Factory",
            "Margin_Volatility"
        ],
        title="Product Cost vs Sales Analysis"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
        "Products with high costs relative to sales and low margins "
        "may need pricing review or cost negotiation."
    )

st.divider()


# ==================================================
# 16. PROFIT CONCENTRATION - PARETO ANALYSIS
# ==================================================

st.markdown("## 5. Profit Concentration – Pareto Analysis")

if not filtered_products.empty:

    pareto = filtered_products.sort_values(
        "Gross_Profit",
        ascending=False
    ).copy()

    pareto_profit = pareto["Gross_Profit"].sum()
    pareto_sales = pareto["Sales"].sum()

    if pareto_profit > 0:

        pareto["Profit_Contribution"] = (
            pareto["Gross_Profit"]
            / pareto_profit * 100
        )

        pareto["Cumulative_Profit"] = (
            pareto["Profit_Contribution"].cumsum()
        )

    else:
        pareto["Profit_Contribution"] = 0
        pareto["Cumulative_Profit"] = 0

    if pareto_sales > 0:

        pareto["Revenue_Contribution"] = (
            pareto["Sales"] / pareto_sales * 100
        )

        pareto["Cumulative_Revenue"] = (
            pareto["Revenue_Contribution"].cumsum()
        )

    else:
        pareto["Revenue_Contribution"] = 0
        pareto["Cumulative_Revenue"] = 0

    fig = px.line(
        pareto,
        x="Product Name",
        y=[
            "Cumulative_Profit",
            "Cumulative_Revenue"
        ],
        markers=True,
        title="Cumulative Profit and Revenue Contribution"
    )

    fig.add_hline(
        y=80,
        line_dash="dash",
        annotation_text="80% Target"
    )

    fig.update_xaxes(tickangle=45)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("### Pareto Product Contribution Table")

    st.dataframe(
        pareto[
            [
                "Product Name",
                "Gross_Profit",
                "Profit_Contribution",
                "Cumulative_Profit",
                "Revenue_Contribution",
                "Cumulative_Revenue"
            ]
        ].round(2),
        use_container_width=True,
        hide_index=True
    )

    def products_to_reach_80(series):

        reached = np.flatnonzero(
            series.to_numpy() >= 80
        )

        return (
            int(reached[0] + 1)
            if len(reached)
            else len(series)
        )

    p1, p2 = st.columns(2)

    p1.metric(
        "Products to Reach 80% Profit",
        products_to_reach_80(
            pareto["Cumulative_Profit"]
        )
    )

    p2.metric(
        "Products to Reach 80% Revenue",
        products_to_reach_80(
            pareto["Cumulative_Revenue"]
        )
    )

st.divider()


# ==================================================
# 17. FACTORY PERFORMANCE ANALYSIS
# ==================================================

st.markdown("## 6. Factory Performance Analysis")

factory_data = (
    filtered_df.groupby(
        "Factory",
        as_index=False
    )
    .agg(
        Sales=("Sales", "sum"),
        Gross_Profit=("Gross Profit", "sum"),
        Cost=("Cost", "sum"),
        Units=("Units", "sum"),
        Orders=("Order ID", "nunique"),
        Product_Count=("Product Name", "nunique")
    )
)

factory_data["Gross_Margin"] = np.where(
    factory_data["Sales"] > 0,
    factory_data["Gross_Profit"]
    / factory_data["Sales"] * 100,
    0
)

factory_data["Profit_Contribution"] = np.where(
    total_profit > 0,
    factory_data["Gross_Profit"]
    / total_profit * 100,
    0
)

factory_data = factory_data.sort_values(
    "Gross_Profit",
    ascending=False
)

if not factory_data.empty:

    fc1, fc2 = st.columns(2)

    with fc1:

        fig = px.bar(
            factory_data,
            x="Factory",
            y="Gross_Profit",
            color="Factory",
            title="Factory-wise Gross Profit"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with fc2:

        fig = px.bar(
            factory_data,
            x="Factory",
            y="Sales",
            color="Factory",
            title="Factory-wise Revenue"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("### Factory Performance Table")

    st.dataframe(
        factory_data.round(2),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Product-to-Factory Mapping")

    mapping_table = (
        filtered_df[
            ["Division", "Product Name", "Factory"]
        ]
        .drop_duplicates()
        .sort_values(
            ["Factory", "Product Name"]
        )
    )

    st.dataframe(
        mapping_table,
        use_container_width=True,
        hide_index=True
    )

unmapped = filtered_df[
    filtered_df["Factory"] == "Unmapped"
]["Product Name"].unique()

if len(unmapped):
    st.warning(
        f"Unmapped products: {list(unmapped)}"
    )
else:
    st.success(
        "All selected products are matched to a factory."
    )

st.divider()


# ==================================================
# 18. FACTORY LOCATION MAP
# ==================================================

st.markdown("## 7. Factory Locations")

factory_map_data = pd.DataFrame([
    {
        "Factory": name,
        "Latitude": details["Latitude"],
        "Longitude": details["Longitude"]
    }
    for name, details in FACTORY_COORDINATES.items()
])

fig = px.scatter_geo(
    factory_map_data,
    lat="Latitude",
    lon="Longitude",
    hover_name="Factory",
    scope="usa",
    projection="albers usa",
    title="Locations of the Five Factories"
)

fig.update_traces(
    marker=dict(size=12)
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.caption(
    "This map displays the supplied factory coordinates. "
    "Customer coordinates are not present in the dataset, "
    "so actual factory-to-customer routes and distances "
    "cannot yet be calculated."
)

st.divider()


# ==================================================
# 19. STATE AND REGION ORDER ANALYSIS
# ==================================================

st.markdown("## 8. State and Region Order Concentration")

region_data = (
    filtered_df.groupby(
        "Region",
        as_index=False
    )
    .agg(
        Total_Orders=("Order ID", "nunique"),
        Sales=("Sales", "sum"),
        Gross_Profit=("Gross Profit", "sum"),
        Units=("Units", "sum"),
        State_Count=("State/Province", "nunique")
    )
)

state_data = (
    filtered_df.groupby(
        ["State/Province", "Region"],
        as_index=False
    )
    .agg(
        Total_Orders=("Order ID", "nunique"),
        Sales=("Sales", "sum"),
        Gross_Profit=("Gross Profit", "sum"),
        Units=("Units", "sum")
    )
    .sort_values(
        "Total_Orders",
        ascending=False
    )
)

g1, g2 = st.columns(2)

with g1:

    fig = px.bar(
        region_data,
        x="Region",
        y="Total_Orders",
        color="Region",
        title="Region-wise Order Volume"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with g2:

    top_states = state_data.head(10).sort_values(
        "Total_Orders"
    )

    fig = px.bar(
        top_states,
        x="Total_Orders",
        y="State/Province",
        color="Region",
        orientation="h",
        title="Top 10 States by Order Volume"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

st.markdown("### Region Performance Table")

st.dataframe(
    region_data.round(2),
    use_container_width=True,
    hide_index=True
)

st.markdown("### State Performance Table")

st.dataframe(
    state_data.round(2),
    use_container_width=True,
    hide_index=True
)

st.caption(
    "High order concentration may indicate operational dependency "
    "or a need for capacity planning. It does not prove traffic congestion."
)

st.divider()


# ==================================================
# 20. SHIPPING DATE DATA QUALITY CHECK
# ==================================================

st.markdown("## 9. Shipping Date Data Quality Check")

if "Ship Date" in df.columns:

    ship_dates = pd.to_datetime(
        df["Ship Date"],
        errors="coerce"
    )

    order_dates = pd.to_datetime(
        df["Order Date"],
        errors="coerce"
    )

    valid_dates = (
        order_dates.notna()
        & ship_dates.notna()
    )

    if valid_dates.any():

        date_check = pd.DataFrame({
            "Order Date": order_dates[valid_dates],
            "Ship Date": ship_dates[valid_dates]
        })

        date_check["Days Between Order and Ship"] = (
            date_check["Ship Date"]
            - date_check["Order Date"]
        ).dt.days

        future_shipments = (
            date_check["Days Between Order and Ship"] < 0
        ).sum()

        st.metric(
            "Records with Ship Date Before Order Date",
            f"{future_shipments:,}"
        )

        st.dataframe(
            date_check[
                "Days Between Order and Ship"
            ]
            .describe()
            .to_frame(
                "Days Between Order and Ship"
            )
            .round(2),
            use_container_width=True
        )

        st.caption(
            "Review unusually large date gaps before using them "
            "as delivery-time or shipping-efficiency measures."
        )

    else:
        st.info(
            "No valid order and ship date pairs are available."
        )

else:
    st.info(
        "Ship Date column is not available."
    )


# ==================================================
# 21. FINAL MESSAGE
# ==================================================

st.success(
    "Nassau Candy Distributor Dashboard loaded successfully!"
)
