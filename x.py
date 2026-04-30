import polars as pl

DEFAULT_DF = pl.DataFrame(
    {
        "Carrier": [
            "DHL",
            "FedEx",
            "UPS",
            "Amazon",
            "DB Schenker",
            "Maersk",
            "Kuehne+Nagel",
        ],
        "Mon": [142, 120, 110, 200, 85, 60, 50],
        "Tue": [138, 125, 115, 210, 90, 65, 55],
        "Wed": [155, 130, 120, 225, 95, 70, 58],
        "Thu": [160, 145, 135, 240, 100, 68, 62],
        "Fri": [170, 155, 148, 260, 105, 72, 65],
        "Sat": [90, 80, 75, 150, 40, 30, 25],
        "Sun": [40, 30, 25, 80, 10, 5, 5],
    }
)
long_df = DEFAULT_DF.unpivot(index="Carrier", variable_name="date", value_name="value")
print(long_df)
