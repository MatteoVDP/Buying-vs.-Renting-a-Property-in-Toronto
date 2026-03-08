
rent_increase = 0.021
inflation = 0.02
average_rent = 3449
mortgage_payment_monthly = 4740.81
own_extra_costs = 1557
rent_extra_costs = 469
TSX_returns = 0.103
monthly_RoR = TSX_returns / 12
periods = 300

median_2025_price = 1090320
median_2050_price = 2288150.38
initial_down_payment = median_2025_price * 0.2 

portfolio = initial_down_payment 

if __name__ == "__main__":

    for month in range(periods):
        portfolio = portfolio * (1 + monthly_RoR)

        if month % 12 == 11: average_rent = average_rent * (1 + rent_increase) 
        rent_extra_costs = rent_extra_costs * (1 + inflation/12)
        own_extra_costs = own_extra_costs * (1 + inflation/12)

        total_rent_cost = average_rent + rent_extra_costs
        total_own_cost = mortgage_payment_monthly + own_extra_costs
        
        diff = total_own_cost - total_rent_cost
        
        portfolio = portfolio + diff

    print(f"FINAL RENT REVENUE (TSX Portfolio): ${portfolio:,.2f}")
    print(f"FINAL OWN REVENUE (House Equity): ${median_2050_price:,.2f}")