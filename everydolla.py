import streamlit as st
import plotly.graph_objects as go
import calendar
from datetime import datetime
from streamlit_option_menu import option_menu

# ---- SETTINGS ----
incomes = ["Salary", "Bonus", "Overtime", "Other Income"]
expenses = ["Mortgage", "Utilities", "WiFi", 
            "Phone", "Health Insurance", "Groceries", 
            "Restaurant", "Date Night", "Gas", 
            "Kailynn's After School", "Kam's After School", 
            "Cheer", "Kam's Daycare", "Auto Insurance", 
            "Lawn Care", "Tahoe Loan", "Counseling", 
            "Calm App", "Child Support"]
savings = ["Emergency Fund", "NOLA", "Investments", 
           "Beater Car", "Vehicle Maintenance", 
           "Christmas", "Kam Fun", "Kailynn Fun", "Gifts",
            "Clothing", "Family Vacation", "Medical", 
            "House Maintenance", "Misc. Housing"]
currency = "USD"
page_title = "Budget Automator"
page_icon = ":money_with_wings:"
layout = "centered"

st.set_page_config(page_title=page_title,page_icon=page_icon,layout=layout)
st.title(page_title + " " + page_icon)

years = [datetime.today().year, datetime.today().year + 1]
months = list(calendar.month_name[1:])

hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""

selected = option_menu(
    menu_title=None,
    options = ["Data Entry", "Data Visualization"],
    icons= ["pencil-fill", "bar-chart-fill"],
    orientation="horizontal",
    )

st. header(f"Data Entry in {currency}")
if selected == "Data Entry":
    with st.form("entry_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        col1.selectbox("Select Month:", months, key="month")
        col2.selectbox("Select Year:", years, key="year")

        "---"
        with st.expander("Income"):
            for income in incomes:
                st.number_input(f"{income}:", min_value=0, format="%i", step=10, key=income)
        with st.expander("Expenses"):
            for expense in expenses:
                st.number_input(f"{expense}:", min_value=0, format="%i", step=10, key=expense)
        with st.expander("Savings"):
            for saving in savings:
                st.number_input(f"{saving}:", min_value=0, format="%i", step=10, key=saving)
        with st.expander("Comments"):
            comment = st.text_area("", placeholder= "Enter your comment here...")

        "---"
        submitted = st.form_submit_button("Save Data")
        if submitted:
            period = str(st.session_state["year"]) + "_" + str(st.session_state["month"])
            incomes = {income: st.session_state[income] for income in incomes}
            expenses = {expense: st.session_state[expense] for expense in expenses}

            st.write(f"incomes: {incomes}")
            st.write(f"expenses: {expenses}")
            st.write(f"savings: {savings}")
            st.success("Data Saved!")

if selected == "Data Visualization":
    st.header("Data Visualization")
    with st.form("saved_periods"):
            period = st.selectbox("Select Period:", ["2023_April"])
            submitted = st.form_submit_button("Plot Period")
            if submitted:
                comment = "Some Comment"
                incomes = {'Salary': 1500, 'Bonus': 150, 'Overtime': 200}
                expenses = {'Mortgage': 500, 'Utilities': 250, 'Groceries': 400}
                savings = {'NOLA': 350, 'Investments': 200}

                total_income = sum(incomes.values())
                total_expense = sum(expenses.values())
                total_saved = sum(savings.values())
                remaining_budget = total_income - total_expense
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Income", f"{total_income}{currency}")
                col2.metric("Total Expense", f"{total_expense}{currency}")
                col3.metric("Total Saved", f"{total_saved}{currency}")
                col4.metric("Remaining Budget", f"{remaining_budget}{currency}")
                st.text(f"Comment: {comment}")

                label = list(incomes.keys()) + ["Total Income"] + list(expenses.keys()) + list(savings.keys())
                source = list(range(len(incomes))) + [len(incomes)] * len(expenses) * len(savings)
                target = [len(incomes)] * len(incomes) + [label.index(expense) for expense in expenses] + [label.index(saving) for saving in savings] 
                value = list(incomes.values()) + list(expenses.values()) + list(savings.values())

                link = dict(source=source, target=target, value=value)
                node = dict(label=label, pad=20, thickness=30, color="#E694FF")
                data = go.Sankey(link=link, node=node)

                fig = go.Figure(data)
                fig.update_layout(margin=dict(l=0, r=0, t=5, b=5))
                st.plotly_chart(fig, use_container_width=True)
