
# --- __main__ demo values ---
average_rent = 3449
median_2025_price = 1090320
median_2050_price = 4299326

# --- Hardcoded assumptions ---
DOWN_PAYMENT_PCT = 0.20
ANNUAL_MORTGAGE_RATE = 0.0432
AMORTIZATION_MONTHS = 300  # 25 years
RENT_INCREASE = 0.021
INFLATION = 0.02
OWN_EXTRA_COSTS = 1557
RENT_EXTRA_COSTS = 469
TSX_RETURNS = 0.103


def calculate_canadian_mortgage(home_price):
    # 1. Calculate the actual loan principal
    principal = home_price * (1 - DOWN_PAYMENT_PCT)

    # 2. Canadian Semi-Annual Compounding Math
    # Converts the annual rate to an effective monthly rate
    monthly_rate = ((1 + (ANNUAL_MORTGAGE_RATE / 2)) ** (2 / 12)) - 1

    # 3. Standard Amortization Formula: PMT = P * [r(1+r)^n] / [(1+r)^n - 1]
    numerator = monthly_rate * ((1 + monthly_rate) ** AMORTIZATION_MONTHS)
    denominator = ((1 + monthly_rate) ** AMORTIZATION_MONTHS) - 1
    monthly_payment = principal * (numerator / denominator)

    return round(monthly_payment, 2)

def calculate_portfolio_return(home_price, monthly_rent):
    initial_portfolio = home_price * DOWN_PAYMENT_PCT
    mortgage_payment_monthly = calculate_canadian_mortgage(home_price)
    monthly_RoR = TSX_RETURNS / 12

    portfolio = initial_portfolio
    average_rent = monthly_rent
    rent_extra_costs = RENT_EXTRA_COSTS
    own_extra_costs = OWN_EXTRA_COSTS

    for month in range(AMORTIZATION_MONTHS):
        portfolio = portfolio * (1 + monthly_RoR)

        if month % 12 == 11: average_rent = average_rent * (1 + RENT_INCREASE)
        rent_extra_costs = rent_extra_costs * (1 + INFLATION / 12)
        own_extra_costs = own_extra_costs * (1 + INFLATION / 12)

        total_rent_cost = average_rent + rent_extra_costs
        total_own_cost = mortgage_payment_monthly + own_extra_costs

        diff = total_own_cost - total_rent_cost

        portfolio = portfolio + diff

    return portfolio


if __name__ == "__main__":

    final_portfolio = calculate_portfolio_return(
        home_price=median_2025_price,
        monthly_rent=average_rent,
    )

    print(f"FINAL RENT REVENUE (TSX Portfolio): ${final_portfolio:,.2f}")
    print(f"FINAL OWN REVENUE (House Equity): ${median_2050_price:,.2f}")