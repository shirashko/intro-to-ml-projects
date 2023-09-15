import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset
    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=["Date"]).dropna().drop_duplicates()
    df = df[df.Temp > -40]  # some examples report a daily temperature of -72 degrees which is not possible if we check
    # the minimal degrees measured in those countries in those years. so we cut this invalid data
    df["DayOfYear"] = df["Date"].dt.dayofyear  # The .dt accessor is used to access the datetime properties of the
    # "Date" column, and the dayofyear attribute is used to extract the day of the year from each datetime value in
    # the "Date" column. This gives a numeric value between 1 and 366
    df["Year"] = df["Year"].astype(str)  # for plotting the dots in a discrete way (later)
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    data = load_data(r"C:\Users\97252\OneDrive - Yezreel Valley College\Desktop\IML.HUJI\datasets\City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    Israel_data = data[data["Country"] == "Israel"]
    px.scatter(Israel_data, x="DayOfYear", y="Temp", color="Year", title='the relation between the day of the year '
                                                                         'and the temperature in Israel').show()
    df_of_monthly_std_temp = Israel_data.groupby(["Month"], as_index=False).agg(std=("Temp", "std"))  # as_index
    # argument adds to the df be the column of the Month as well instead of making the month into the index (id).
    # agg(std=("Temp", "std")) is passed to agg function as a dictionary with a single key-value pair. The key is "std",
    # which represents the name of the new column to create, and the value is a tuple ("Temp", "std") representing the
    # column to aggregate, and the aggregation function to apply (in this case, the std).

    px.bar(df_of_monthly_std_temp, title="The std of daily temperature in each month over the years", x="Month",
           y="std", text_auto=True).show()

    # Question 3 - Exploring differences between countries
    d = data.groupby(["Country", "Month"], as_index=False).agg(std=("Temp", "std"), average=("Temp", "mean"))
    px.line(d, x="Month", y="average", error_y="std", color="Country").update_layout(
        title="the average monthly temperature in each country over the years", xaxis_title="month",
        yaxis_title="average temperature").show()

    # Question 4 - Fitting model for different values of `k`
    # predict by the day of the year the daily temperature on the Israel data set
    train_X, train_y, test_X, test_y = split_train_test(Israel_data["DayOfYear"], Israel_data["Temp"])
    k_losses = []
    degrees = list(range(1, 11))
    for k in degrees:
        model = PolynomialFitting(k)
        model.fit(train_X.values, train_y.values)
        loss = round(model.loss(test_X.values, test_y.values), 2)
        k_losses.append(loss)
        print(f"loss of a model with k = {k} over the test set is {loss}")
    k_df = pd.DataFrame(dict(k=degrees, loss=k_losses))
    k_df["k"] = k_df["k"].astype(str)
    px.bar(k_df, x="k", y="loss", title="the test error for degrees in range 1-10", text="loss", color="k").show()
    # the text argument tells Plotly to display the k values on top of each bar in the bar chart.

    # Question 5 - Evaluating fitted model on different countries
    best_k = degrees[k_losses.index(min(k_losses))]
    model = PolynomialFitting(best_k)
    model.fit(Israel_data["DayOfYear"].values, Israel_data["Temp"].values)
    data_without_Israel = data[data["Country"] != "Israel"].groupby("Country")
    countries, countries_loss = [], []
    for country, group in data_without_Israel:
        countries.append(country)
        loss = round(model.loss(group["DayOfYear"], group["Temp"]), 2)  # todo round?
        countries_loss.append(loss)
    d = pd.DataFrame(dict(Countries=countries, loss=countries_loss))
    px.bar(d, x="Countries", y="loss", title="Israel trained model’s error over each of the other countries",
           text="loss", color="Countries").show()
