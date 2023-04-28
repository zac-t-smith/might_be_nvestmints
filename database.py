from deta import Deta

DETA_KEY = "3Riqwp23_cM4gKmRjzPxTRxFqkAkQsrmG3444uY84"

# Initialize with a project key
deta = Deta(DETA_KEY)

# This is how to create/connect a database
db = deta.Base("monthly_reports")


def insert_period(period, incomes, expenses, savings, comment):
    """Returns the report on a successful creation, otherwise raises an error"""
    return db.put({"key": period, "incomes": incomes, "expenses": expenses, "savings": savings, "comment": comment})


def fetch_all_periods():
    """Returns a dict of all periods"""
    res = db.fetch()
    return res.items


def get_period(period):
    """If not found, the function will return None"""
    return db.get(period)